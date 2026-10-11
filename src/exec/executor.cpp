#include "exec/executor.h"

#include "exec/block.h"
#include "exec/register_file.h"
#include "mem/atomics.h"
#include "mem/coalescer.h"
#include "ptx/lexer.h"
#include "ptx/token.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudasim::exec {

namespace {

struct FlatInstruction {
  std::string opcode;
  std::string predicate;
  std::string operands;
  std::vector<cudasim::ptx::Token> operand_tokens;
  std::vector<cudasim::ptx::Token> predicate_tokens;
};

std::vector<cudasim::ptx::Token> tokenize(const std::string &text) {
  cudasim::ptx::Lexer lexer(text);
  std::vector<cudasim::ptx::Token> tokens;
  for (;;) {
    cudasim::ptx::Token token = lexer.next();
    if (token.kind == cudasim::ptx::TokenKind::End) {
      break;
    }
    tokens.push_back(std::move(token));
  }
  return tokens;
}

struct RegisterRef {
  std::size_t index = 0;
  int size = 4;
};

std::optional<RegisterRef>
parse_register(const std::vector<cudasim::ptx::Token> &tokens,
               std::size_t &pos) {
  while (pos < tokens.size() &&
         tokens[pos].kind == cudasim::ptx::TokenKind::Comma) {
    ++pos;
  }
  if (pos < tokens.size() &&
      tokens[pos].kind == cudasim::ptx::TokenKind::LBrace) {
    ++pos;
  }
  if (pos >= tokens.size() ||
      tokens[pos].kind != cudasim::ptx::TokenKind::Percent) {
    return std::nullopt;
  }
  ++pos;
  if (pos >= tokens.size() ||
      tokens[pos].kind != cudasim::ptx::TokenKind::Identifier) {
    return std::nullopt;
  }

  const std::string name = tokens[pos].text;
  ++pos;

  char bank = 'r';
  std::size_t number_start = 0;
  if (name.size() >= 2 && name[0] == 'r' && name[1] == 'd') {
    bank = 'd';
    number_start = 2;
  } else if (name.size() >= 2 && name[0] == 'f') {
    bank = 'f';
    number_start = 1;
  } else if (!name.empty() && name[0] == 'r') {
    bank = 'r';
    number_start = 1;
  } else {
    return std::nullopt;
  }

  std::size_t index = 0;
  for (std::size_t i = number_start; i < name.size(); ++i) {
    index = index * 10 + static_cast<std::size_t>(name[i] - '0');
  }

  RegisterRef ref;
  std::size_t base = 0;
  // Keep r/rd/f register banks separated so large r-register indices such as
  // r716 do not alias rd1 or f1.
  if (bank == 'd') {
    base = kRegisterDBankBase;
  } else if (bank == 'f') {
    base = kRegisterFBankBase;
  }
  if (index >= kRegisterBankSize) {
    return std::nullopt;
  }
  ref.index = base + index;
  ref.size = (bank == 'd' || bank == 'f') ? 8 : 4;
  if (pos < tokens.size() &&
      tokens[pos].kind == cudasim::ptx::TokenKind::RBrace) {
    ++pos;
  }
  return ref;
}

std::uint32_t special_register_u32(const std::string &name,
                                   const LaunchConfig &config,
                                   const BlockInfo &block,
                                   std::size_t lane_in_block,
                                   std::size_t lane) {
  const unsigned int block_id_x = block.index[0];
  const unsigned int block_dim_x = config.block[0];
  const unsigned int grid_dim_x = config.grid[0];
  const unsigned int block_id_y = block.index[1];
  const unsigned int block_id_z = block.index[2];
  const unsigned int block_dim_y = config.block[1];
  const unsigned int block_dim_z = config.block[2];
  const unsigned int grid_dim_y = config.grid[1];
  const unsigned int grid_dim_z = config.grid[2];
  const std::size_t tid_x = lane_in_block % block_dim_x;
  // Convert the linear lane id back to the 3D thread/block coordinates used
  // by PTX special registers.
  const std::size_t tid_y =
      (lane_in_block / block_dim_x) % std::max(1U, block_dim_y);
  const std::size_t tid_z =
      lane_in_block /
      (static_cast<std::size_t>(block_dim_x) * std::max(1U, block_dim_y));
  if (name == "%tid.x") {
    return static_cast<std::uint32_t>(tid_x);
  }
  if (name == "%tid.y") {
    return static_cast<std::uint32_t>(tid_y);
  }
  if (name == "%tid.z") {
    return static_cast<std::uint32_t>(tid_z);
  }
  if (name == "%ntid.x") {
    return block_dim_x;
  }
  if (name == "%ntid.y") {
    return block_dim_y;
  }
  if (name == "%ntid.z") {
    return block_dim_z;
  }
  if (name == "%ctaid.x") {
    return block_id_x;
  }
  if (name == "%ctaid.y") {
    return block_id_y;
  }
  if (name == "%ctaid.z") {
    return block_id_z;
  }
  if (name == "%nctaid.x") {
    return grid_dim_x;
  }
  if (name == "%nctaid.y") {
    return grid_dim_y;
  }
  if (name == "%nctaid.z") {
    return grid_dim_z;
  }
  if (name == "%laneid") {
    return static_cast<std::uint32_t>(lane % 32);
  }
  if (name == "%warpid") {
    return static_cast<std::uint32_t>(lane / 32);
  }
  return 0;
}

bool parse_predicate(const std::vector<cudasim::ptx::Token> &tokens,
                     const RegisterFile &registers, std::size_t lane) {
  if (tokens.empty()) {
    return true;
  }

  bool negate = false;
  std::size_t pos = 0;
  if (tokens[pos].kind == cudasim::ptx::TokenKind::Exclaim) {
    negate = true;
    ++pos;
  }
  if (pos >= tokens.size() ||
      tokens[pos].kind != cudasim::ptx::TokenKind::Percent) {
    return true;
  }
  ++pos;
  if (pos >= tokens.size() ||
      tokens[pos].kind != cudasim::ptx::TokenKind::Identifier) {
    return true;
  }
  const std::string name = tokens[pos].text;
  std::size_t index = 0;
  for (std::size_t i = 1; i < name.size(); ++i) {
    index = index * 10 + static_cast<std::size_t>(name[i] - '0');
  }
  if (index >= registers.predicates.size()) {
    return false;
  }
  const bool value = registers.predicates[index][lane];
  return negate ? !value : value;
}

std::optional<std::uint32_t>
parse_u32_operand(const std::vector<cudasim::ptx::Token> &tokens,
                  std::size_t &pos, const RegisterFile &registers,
                  std::size_t lane) {
  while (pos < tokens.size() &&
         tokens[pos].kind == cudasim::ptx::TokenKind::Comma) {
    ++pos;
  }
  if (pos >= tokens.size()) {
    return std::nullopt;
  }
  bool negative = false;
  if (tokens[pos].kind == cudasim::ptx::TokenKind::Minus) {
    negative = true;
    ++pos;
  }
  if (tokens[pos].kind == cudasim::ptx::TokenKind::Number) {
    const std::uint32_t value =
        static_cast<std::uint32_t>(std::stoul(tokens[pos].text));
    ++pos;
    return negative
               ? static_cast<std::uint32_t>(-static_cast<std::int32_t>(value))
               : value;
  }
  auto reg = parse_register(tokens, pos);
  if (!reg) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(registers.values[reg->index][lane]);
}

std::optional<float> parse_ptx_f32_immediate(const std::string &text) {
  if (text.size() == 10 && text[0] == '0' && text[1] == 'f') {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(std::stoul(text.substr(2), nullptr, 16));
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  return std::stof(text);
}

std::optional<float>
parse_f32_operand(const std::vector<cudasim::ptx::Token> &tokens,
                  std::size_t &pos, const RegisterFile &registers,
                  std::size_t lane) {
  while (pos < tokens.size() &&
         tokens[pos].kind == cudasim::ptx::TokenKind::Comma) {
    ++pos;
  }
  if (pos >= tokens.size()) {
    return std::nullopt;
  }
  if (tokens[pos].kind == cudasim::ptx::TokenKind::Number) {
    const std::string text = tokens[pos].text;
    ++pos;
    return parse_ptx_f32_immediate(text);
  }
  auto reg = parse_register(tokens, pos);
  if (!reg) {
    return std::nullopt;
  }
  float value = 0.0F;
  std::memcpy(&value, &registers.values[reg->index][lane], sizeof(value));
  return value;
}

std::size_t
find_label(const std::unordered_map<std::string, std::size_t> &labels,
           const std::string &label) {
  const auto it = labels.find(label);
  return it == labels.end() ? 0 : it->second;
}

std::size_t type_size(const std::string &type) {
  if (type == "b8" || type == "u8" || type == "s8") {
    return 1;
  }
  if (type == "b16" || type == "u16" || type == "s16" || type == "f16") {
    return 2;
  }
  if (type == "b32" || type == "u32" || type == "s32" || type == "f32") {
    return 4;
  }
  if (type == "b64" || type == "u64" || type == "s64" || type == "f64") {
    return 8;
  }
  return 4;
}

bool coalesced_global_f32(const FlatInstruction &instruction,
                          RegisterFile &registers,
                          const std::vector<std::size_t> &lanes, bool store) {
  std::size_t pos = 0;

  if (!store) {
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    if (!address) {
      return false;
    }
    // PTX memory operands may have signed offsets such as [%rd+ -12].
    std::int64_t immediate = 0;
    bool negative = false;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Minus) {
      negative = true;
      ++pos;
    } else if (pos < instruction.operand_tokens.size() &&
               instruction.operand_tokens[pos].kind ==
                   cudasim::ptx::TokenKind::Plus) {
      ++pos;
      if (pos < instruction.operand_tokens.size() &&
          instruction.operand_tokens[pos].kind ==
              cudasim::ptx::TokenKind::Minus) {
        negative = true;
        ++pos;
      }
    }
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      immediate = static_cast<std::int64_t>(
          std::stoll(instruction.operand_tokens[pos].text));
      if (negative) {
        immediate = -immediate;
      }
      ++pos;
    }

    std::vector<std::size_t> memory_lanes;
    memory_lanes.reserve(lanes.size());
    for (const std::size_t lane : lanes) {
      if (parse_predicate(instruction.predicate_tokens, registers, lane)) {
        memory_lanes.push_back(lane);
      }
    }

    std::vector<const void *> addresses;
    addresses.reserve(memory_lanes.size());
    for (const std::size_t lane : memory_lanes) {
      addresses.push_back(reinterpret_cast<const void *>(
          static_cast<std::uintptr_t>(registers.values[address->index][lane] +
                                      static_cast<std::uint64_t>(immediate))));
    }
    std::vector<std::uint32_t> values;
    cudasim::mem::coalesced_read_u32_pointers(addresses, values);
    for (std::size_t i = 0; i < memory_lanes.size(); ++i) {
      registers.values[dst->index][memory_lanes[i]] = values[i];
    }
    return true;
  }

  while (pos < instruction.operand_tokens.size() &&
         instruction.operand_tokens[pos].kind !=
             cudasim::ptx::TokenKind::LBracket) {
    ++pos;
  }
  if (pos >= instruction.operand_tokens.size()) {
    return false;
  }
  ++pos;
  auto address = parse_register(instruction.operand_tokens, pos);
  std::int64_t immediate = 0;
  bool negative = false;
  if (pos < instruction.operand_tokens.size() &&
      instruction.operand_tokens[pos].kind == cudasim::ptx::TokenKind::Minus) {
    negative = true;
    ++pos;
  } else if (pos < instruction.operand_tokens.size() &&
             instruction.operand_tokens[pos].kind ==
                 cudasim::ptx::TokenKind::Plus) {
    ++pos;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Minus) {
      negative = true;
      ++pos;
    }
  }
  if (pos < instruction.operand_tokens.size() &&
      instruction.operand_tokens[pos].kind == cudasim::ptx::TokenKind::Number) {
    immediate = static_cast<std::int64_t>(
        std::stoll(instruction.operand_tokens[pos].text));
    if (negative) {
      immediate = -immediate;
    }
    ++pos;
  }
  while (pos < instruction.operand_tokens.size() &&
         instruction.operand_tokens[pos].kind !=
             cudasim::ptx::TokenKind::Percent) {
    ++pos;
  }
  auto value = parse_register(instruction.operand_tokens, pos);
  if (!address || !value) {
    return false;
  }

  std::vector<std::size_t> memory_lanes;
  memory_lanes.reserve(lanes.size());
  for (const std::size_t lane : lanes) {
    if (parse_predicate(instruction.predicate_tokens, registers, lane)) {
      memory_lanes.push_back(lane);
    }
  }

  std::vector<void *> addresses;
  std::vector<std::uint32_t> values;
  addresses.reserve(memory_lanes.size());
  values.reserve(memory_lanes.size());
  for (const std::size_t lane : memory_lanes) {
    addresses.push_back(reinterpret_cast<void *>(
        static_cast<std::uintptr_t>(registers.values[address->index][lane] +
                                    static_cast<std::uint64_t>(immediate))));
    values.push_back(
        static_cast<std::uint32_t>(registers.values[value->index][lane]));
  }
  cudasim::mem::coalesced_write_u32_pointers(addresses, values);
  return true;
}

bool execute_instruction(
    const FlatInstruction &instruction, const LaunchConfig &config,
    const BlockInfo &block, RegisterFile &registers, std::size_t lane,
    std::size_t lane_in_block, const std::vector<std::uint8_t> &arguments,
    const std::unordered_map<std::string, std::size_t> &parameter_offsets,
    std::vector<std::uint8_t> &shared_memory,
    const std::unordered_map<std::string, std::size_t> &shared_offsets,
    std::size_t &next_pc, std::size_t pc,
    const std::unordered_map<std::string, std::size_t> &labels) {
  const std::string &opcode = instruction.opcode;

  if (opcode == "bra" || opcode == "bra.uni") {
    if (!parse_predicate(instruction.predicate_tokens, registers, lane)) {
      next_pc = pc + 1;
      return true;
    }
    if (instruction.operand_tokens.empty()) {
      next_pc = pc + 1;
      return true;
    }
    if (instruction.operand_tokens[0].kind == cudasim::ptx::TokenKind::Dollar) {
      std::string label;
      if (instruction.operand_tokens.size() > 1) {
        label = instruction.operand_tokens[1].text;
      }
      next_pc = find_label(labels, label);
    } else {
      next_pc = pc + 1;
    }
    return true;
  }

  if (!parse_predicate(instruction.predicate_tokens, registers, lane)) {
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "ret" || opcode == "exit") {
    return false;
  }

  if (opcode == "ld.param.u64" || opcode == "ld.param.u32" ||
      opcode == "ld.param.f32" || opcode == "ld.param.b64" ||
      opcode == "ld.param.b32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst || pos >= instruction.operand_tokens.size()) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    const std::string param_name = instruction.operand_tokens[pos].text;
    const auto it = parameter_offsets.find(param_name);
    if (it == parameter_offsets.end()) {
      return false;
    }
    const std::size_t offset = it->second;
    std::uint64_t value = 0;
    const std::size_t bytes = std::min<std::size_t>(
        static_cast<std::size_t>(dst->size), arguments.size() - offset);
    std::memcpy(&value, arguments.data() + offset, bytes);
    registers.values[dst->index][lane] = value;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mov.u32" || opcode == "mov.b32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    bool negative_immediate = false;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Minus) {
      negative_immediate = true;
      ++pos;
    }
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      const std::string text = instruction.operand_tokens[pos].text;
      const auto value = static_cast<std::uint32_t>(
          text.size() == 10 && text[0] == '0' && text[1] == 'f'
              ? std::stoul(text.substr(2), nullptr, 16)
              : std::stoul(text, nullptr, 0));
      registers.values[dst->index][lane] =
          negative_immediate
              ? static_cast<std::uint32_t>(-static_cast<std::int32_t>(value))
              : value;
      next_pc = pc + 1;
      return true;
    }
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Identifier) {
      const std::string name = instruction.operand_tokens[pos].text;
      const auto shared_it = shared_offsets.find(name);
      if (shared_it != shared_offsets.end()) {
        registers.values[dst->index][lane] =
            static_cast<std::uint32_t>(shared_it->second);
        next_pc = pc + 1;
        return true;
      }
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::Percent) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    const std::size_t reg_start = pos;
    std::size_t ignored = pos;
    auto src = parse_register(instruction.operand_tokens, ignored);
    if (src) {
      registers.values[dst->index][lane] =
          static_cast<std::uint32_t>(registers.values[src->index][lane]);
    } else {
      std::string special;
      for (std::size_t i = reg_start; i < instruction.operand_tokens.size();
           ++i) {
        special += instruction.operand_tokens[i].text;
      }
      registers.values[dst->index][lane] =
          special_register_u32(special, config, block, lane_in_block, lane);
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mad.lo.s32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto c =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b || !c) {
      return false;
    }
    const auto av = static_cast<std::int32_t>(*a);
    const auto bv = static_cast<std::int32_t>(*b);
    const auto cv = static_cast<std::int32_t>(*c);
    const auto result = static_cast<std::uint32_t>(av * bv + cv);
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "bfe.s32" || opcode == "bfe.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto c =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b || !c) {
      return false;
    }
    const std::uint32_t src = *a;
    const std::uint32_t bit = *b & 31U;
    const std::uint32_t width = *c;
    std::uint32_t result = 0;
    if (width > 0) {
      if (width >= 32) {
        result = opcode == "bfe.s32"
                     ? static_cast<std::uint32_t>(
                           static_cast<std::int32_t>(src) >> bit)
                     : src >> bit;
      } else {
        const std::uint32_t mask = (1U << width) - 1U;
        result = (src >> bit) & mask;
        if (opcode == "bfe.s32" && (result & (1U << (width - 1))) != 0) {
          result |= ~mask;
        }
      }
    }
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "or.pred" || opcode == "and.pred" || opcode == "not.pred") {
    std::size_t pos = 0;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string dst_name = instruction.operand_tokens[pos++].text;
    std::size_t dst_index = 0;
    for (std::size_t i = 1; i < dst_name.size(); ++i) {
      dst_index = dst_index * 10 + static_cast<std::size_t>(dst_name[i] - '0');
    }

    auto read_pred = [&](std::size_t &p) -> std::optional<bool> {
      while (p < instruction.operand_tokens.size() &&
             instruction.operand_tokens[p].kind ==
                 cudasim::ptx::TokenKind::Comma) {
        ++p;
      }
      if (p >= instruction.operand_tokens.size() ||
          instruction.operand_tokens[p].kind !=
              cudasim::ptx::TokenKind::Percent) {
        return std::nullopt;
      }
      ++p;
      const std::string name = instruction.operand_tokens[p++].text;
      std::size_t index = 0;
      for (std::size_t i = 1; i < name.size(); ++i) {
        index = index * 10 + static_cast<std::size_t>(name[i] - '0');
      }
      if (index >= registers.predicates.size()) {
        return std::nullopt;
      }
      return registers.predicates[index][lane];
    };

    const auto a = read_pred(pos);
    std::optional<bool> b;
    if (opcode != "not.pred") {
      b = read_pred(pos);
    }
    if (!a || (opcode != "not.pred" && !b) ||
        dst_index >= registers.predicates.size()) {
      return false;
    }
    if (opcode == "or.pred") {
      registers.predicates[dst_index][lane] = *a || *b;
    } else if (opcode == "and.pred") {
      registers.predicates[dst_index][lane] = *a && *b;
    } else {
      registers.predicates[dst_index][lane] = !*a;
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "setp.ge.s32") {
    std::size_t pos = 0;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    if (!a || !b || pred_index >= registers.predicates.size()) {
      return false;
    }
    const auto av = static_cast<std::int32_t>(registers.values[a->index][lane]);
    const auto bv = static_cast<std::int32_t>(registers.values[b->index][lane]);
    registers.predicates[pred_index][lane] = av >= bv;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "setp.ge.f32" || opcode == "setp.gt.f32" ||
      opcode == "setp.lt.f32" || opcode == "setp.le.f32" ||
      opcode == "setp.eq.f32" || opcode == "setp.geu.f32" ||
      opcode == "setp.gtu.f32" || opcode == "setp.leu.f32" ||
      opcode == "setp.ltu.f32" || opcode == "setp.neu.f32") {
    std::size_t pos = 0;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!a || !b || pred_index >= registers.predicates.size()) {
      return false;
    }
    const bool unordered = std::isnan(*a) || std::isnan(*b);
    if (opcode == "setp.ge.f32") {
      registers.predicates[pred_index][lane] = *a >= *b;
    } else if (opcode == "setp.gt.f32") {
      registers.predicates[pred_index][lane] = *a > *b;
    } else if (opcode == "setp.lt.f32") {
      registers.predicates[pred_index][lane] = *a < *b;
    } else if (opcode == "setp.le.f32") {
      registers.predicates[pred_index][lane] = *a <= *b;
    } else if (opcode == "setp.eq.f32") {
      registers.predicates[pred_index][lane] = *a == *b;
    } else if (opcode == "setp.geu.f32") {
      registers.predicates[pred_index][lane] = *a >= *b || unordered;
    } else if (opcode == "setp.gtu.f32") {
      registers.predicates[pred_index][lane] = *a > *b || unordered;
    } else if (opcode == "setp.leu.f32") {
      registers.predicates[pred_index][lane] = *a <= *b || unordered;
    } else if (opcode == "setp.ltu.f32") {
      registers.predicates[pred_index][lane] = *a < *b || unordered;
    } else {
      registers.predicates[pred_index][lane] = *a != *b || unordered;
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "selp.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    if (!dst || !a || !b || pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    if (pred_index >= registers.predicates.size()) {
      return false;
    }
    const float result = registers.predicates[pred_index][lane] ? *a : *b;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "selp.b64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    if (!dst || !a || !b || pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    if (pred_index >= registers.predicates.size()) {
      return false;
    }
    registers.values[dst->index][lane] = registers.predicates[pred_index][lane]
                                             ? registers.values[a->index][lane]
                                             : registers.values[b->index][lane];
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "selp.b32" || opcode == "selp.u32" || opcode == "selp.s32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    if (!dst || !a || !b || pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    if (pred_index >= registers.predicates.size()) {
      return false;
    }
    const std::uint32_t result =
        registers.predicates[pred_index][lane] ? *a : *b;
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "copysign.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b) {
      return false;
    }
    const float result = std::copysign(*b, *a);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "cvta.to.global.u64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src = parse_register(instruction.operand_tokens, pos);
    if (!dst || !src) {
      return false;
    }
    registers.values[dst->index][lane] = registers.values[src->index][lane];
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mul.wide.s32" || opcode == "mul.wide.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    if (!dst || !a || pos >= instruction.operand_tokens.size()) {
      return false;
    }
    std::uint64_t av = 0;
    std::uint64_t bv = 0;
    if (opcode == "mul.wide.s32") {
      av = static_cast<std::uint64_t>(static_cast<std::int64_t>(
          static_cast<std::int32_t>(registers.values[a->index][lane])));
    } else {
      av = static_cast<std::uint32_t>(registers.values[a->index][lane]);
    }
    if (instruction.operand_tokens[pos].kind ==
        cudasim::ptx::TokenKind::Number) {
      if (opcode == "mul.wide.s32") {
        bv = static_cast<std::uint64_t>(static_cast<std::int64_t>(
            std::stoll(instruction.operand_tokens[pos].text)));
      } else {
        bv = static_cast<std::uint64_t>(
            std::stoull(instruction.operand_tokens[pos].text));
      }
    } else {
      auto b = parse_register(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      if (opcode == "mul.wide.s32") {
        bv = static_cast<std::uint64_t>(static_cast<std::int64_t>(
            static_cast<std::int32_t>(registers.values[b->index][lane])));
      } else {
        bv = static_cast<std::uint32_t>(registers.values[b->index][lane]);
      }
    }
    registers.values[dst->index][lane] = av * bv;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mad.wide.s32" || opcode == "mad.wide.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto c = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c) {
      return false;
    }
    const bool is_signed = opcode == "mad.wide.s32";
    const std::uint64_t av =
        is_signed
            ? static_cast<std::uint64_t>(static_cast<std::int64_t>(
                  static_cast<std::int32_t>(registers.values[a->index][lane])))
            : static_cast<std::uint64_t>(
                  static_cast<std::uint32_t>(registers.values[a->index][lane]));
    const std::uint64_t bv =
        is_signed ? static_cast<std::uint64_t>(static_cast<std::int64_t>(
                        static_cast<std::int32_t>(*b)))
                  : static_cast<std::uint64_t>(*b);
    const std::uint64_t cv = registers.values[c->index][lane];
    registers.values[dst->index][lane] = av * bv + cv;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "add.s64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    std::uint64_t bv = 0;
    bool negative = false;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Minus) {
      negative = true;
      ++pos;
    }
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      bv = static_cast<std::uint64_t>(
          std::stoll(instruction.operand_tokens[pos].text));
      if (negative) {
        bv = static_cast<std::uint64_t>(-static_cast<std::int64_t>(bv));
      }
    } else {
      auto b = parse_register(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      bv = registers.values[b->index][lane];
    }
    registers.values[dst->index][lane] = registers.values[a->index][lane] + bv;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "or.b64" || opcode == "and.b64" || opcode == "xor.b64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    std::uint64_t bv = 0;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      bv = static_cast<std::uint64_t>(
          std::stoull(instruction.operand_tokens[pos].text));
    } else {
      auto b = parse_register(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      bv = registers.values[b->index][lane];
    }
    const std::uint64_t av = registers.values[a->index][lane];
    std::uint64_t result = 0;
    if (opcode == "or.b64") {
      result = av | bv;
    } else if (opcode == "and.b64") {
      result = av & bv;
    } else {
      result = av ^ bv;
    }
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "ld.global.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    if (!address) {
      return false;
    }
    std::size_t immediate = 0;
    if (pos + 1 < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind == cudasim::ptx::TokenKind::Plus &&
        instruction.operand_tokens[pos + 1].kind ==
            cudasim::ptx::TokenKind::Number) {
      ++pos;
      immediate = static_cast<std::size_t>(
          std::stoull(instruction.operand_tokens[pos].text));
      ++pos;
    }
    const auto *ptr = reinterpret_cast<const float *>(
        registers.values[address->index][lane] + immediate);
    std::uint32_t bits = 0;
    std::memcpy(&bits, ptr, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "ld.global.u32" || opcode == "ld.global.b32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    if (!address) {
      return false;
    }
    std::size_t immediate = 0;
    if (pos + 1 < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind == cudasim::ptx::TokenKind::Plus &&
        instruction.operand_tokens[pos + 1].kind ==
            cudasim::ptx::TokenKind::Number) {
      ++pos;
      immediate = static_cast<std::size_t>(
          std::stoull(instruction.operand_tokens[pos].text));
      ++pos;
    }
    const auto *ptr = reinterpret_cast<const std::uint32_t *>(
        registers.values[address->index][lane] + immediate);
    registers.values[dst->index][lane] = *ptr;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "add.f32" || opcode == "add.rn.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a) {
      return false;
    }
    float av = 0.0F;
    float bv = 0.0F;
    std::memcpy(&av, &registers.values[a->index][lane], sizeof(av));
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      const std::string immediate = instruction.operand_tokens[pos].text;
      bv = parse_ptx_f32_immediate(immediate).value_or(0.0F);
    } else if (b) {
      std::memcpy(&bv, &registers.values[b->index][lane], sizeof(bv));
    } else {
      return false;
    }
    const float result = av + bv;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "max.f32" || opcode == "min.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b) {
      return false;
    }
    const float result =
        opcode == "max.f32" ? std::fmax(*a, *b) : std::fmin(*a, *b);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mul.f32" || opcode == "mul.rn.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b) {
      return false;
    }
    const float result = *a * *b;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "fma.rn.f32" || opcode == "fma.rm.f32" ||
      opcode == "fma.rz.f32" || opcode == "fma.rn.ftz.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    auto c =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b || !c) {
      return false;
    }
    const float result = std::fmaf(*a, *b, *c);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "st.global.f32") {
    std::size_t pos = 0;
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    if (!address) {
      return false;
    }
    std::size_t immediate = 0;
    if (pos + 1 < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind == cudasim::ptx::TokenKind::Plus &&
        instruction.operand_tokens[pos + 1].kind ==
            cudasim::ptx::TokenKind::Number) {
      ++pos;
      immediate = static_cast<std::size_t>(
          std::stoull(instruction.operand_tokens[pos].text));
      ++pos;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::Percent) {
      ++pos;
    }
    auto value = parse_register(instruction.operand_tokens, pos);
    if (!value) {
      return false;
    }
    auto *ptr = reinterpret_cast<float *>(
        registers.values[address->index][lane] + immediate);
    float fv = 0.0F;
    std::memcpy(&fv, &registers.values[value->index][lane], sizeof(fv));
    *ptr = fv;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "st.global.u32" || opcode == "st.global.b32") {
    std::size_t pos = 0;
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    if (!address) {
      return false;
    }
    std::size_t immediate = 0;
    if (pos + 1 < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind == cudasim::ptx::TokenKind::Plus &&
        instruction.operand_tokens[pos + 1].kind ==
            cudasim::ptx::TokenKind::Number) {
      ++pos;
      immediate = static_cast<std::size_t>(
          std::stoull(instruction.operand_tokens[pos].text));
      ++pos;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::Percent) {
      ++pos;
    }
    auto value = parse_register(instruction.operand_tokens, pos);
    if (!value) {
      return false;
    }
    auto *ptr = reinterpret_cast<std::uint32_t *>(
        registers.values[address->index][lane] + immediate);
    *ptr = static_cast<std::uint32_t>(registers.values[value->index][lane]);
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "shl.b64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src = parse_register(instruction.operand_tokens, pos);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    if (!dst || !src || pos >= instruction.operand_tokens.size()) {
      return false;
    }
    std::uint64_t amount = 0;
    if (instruction.operand_tokens[pos].kind ==
        cudasim::ptx::TokenKind::Number) {
      amount = static_cast<std::uint64_t>(
          std::stoull(instruction.operand_tokens[pos].text));
    } else {
      auto amount_reg = parse_register(instruction.operand_tokens, pos);
      if (!amount_reg) {
        return false;
      }
      amount = registers.values[amount_reg->index][lane];
    }
    registers.values[dst->index][lane] = registers.values[src->index][lane]
                                         << amount;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "shl.b32" || opcode == "shr.s32" || opcode == "shr.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src = parse_register(instruction.operand_tokens, pos);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    if (!dst || !src || pos >= instruction.operand_tokens.size()) {
      return false;
    }
    std::uint32_t amount = 0;
    if (instruction.operand_tokens[pos].kind ==
        cudasim::ptx::TokenKind::Number) {
      amount = static_cast<std::uint32_t>(
          std::stoul(instruction.operand_tokens[pos].text));
    } else {
      auto amount_reg = parse_register(instruction.operand_tokens, pos);
      if (!amount_reg) {
        return false;
      }
      amount =
          static_cast<std::uint32_t>(registers.values[amount_reg->index][lane]);
    }
    const std::uint32_t value =
        static_cast<std::uint32_t>(registers.values[src->index][lane]);
    std::uint32_t result = 0;
    if (opcode == "shl.b32") {
      result = value << amount;
    } else if (opcode == "shr.u32") {
      result = value >> amount;
    } else {
      result = static_cast<std::uint32_t>(static_cast<std::int32_t>(value) >>
                                          amount);
    }
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "bar.sync") {
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "setp.ne.s32" || opcode == "setp.ne.u32") {
    std::size_t pos = 0;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!a || !b || pred_index >= registers.predicates.size()) {
      return false;
    }
    registers.predicates[pred_index][lane] = *a != *b;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "st.shared.f32") {
    std::size_t pos = 0;
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::Percent) {
      ++pos;
    }
    auto value = parse_register(instruction.operand_tokens, pos);
    if (!address || !value) {
      return false;
    }
    const std::size_t byte_offset =
        static_cast<std::size_t>(registers.values[address->index][lane]);
    if (byte_offset + sizeof(float) > shared_memory.size()) {
      return false;
    }
    float fv = 0.0F;
    std::memcpy(&fv, &registers.values[value->index][lane], sizeof(fv));
    std::memcpy(shared_memory.data() + byte_offset, &fv, sizeof(fv));
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "ld.shared.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;

    std::size_t byte_offset = 0;
    bool has_address = false;
    if (auto address = parse_register(instruction.operand_tokens, pos)) {
      byte_offset =
          static_cast<std::size_t>(registers.values[address->index][lane]);
      has_address = true;
      while (pos < instruction.operand_tokens.size() &&
             instruction.operand_tokens[pos].kind !=
                 cudasim::ptx::TokenKind::RBracket) {
        if (instruction.operand_tokens[pos].kind ==
                cudasim::ptx::TokenKind::Plus &&
            pos + 1 < instruction.operand_tokens.size()) {
          ++pos;
          if (instruction.operand_tokens[pos].kind ==
              cudasim::ptx::TokenKind::Number) {
            byte_offset += static_cast<std::size_t>(
                std::stoull(instruction.operand_tokens[pos].text));
          }
        }
        ++pos;
      }
    } else if (pos < instruction.operand_tokens.size() &&
               instruction.operand_tokens[pos].kind ==
                   cudasim::ptx::TokenKind::Identifier) {
      const std::string name = instruction.operand_tokens[pos].text;
      const auto shared_it = shared_offsets.find(name);
      if (shared_it != shared_offsets.end()) {
        byte_offset = shared_it->second;
        has_address = true;
      }
      ++pos;
      while (pos < instruction.operand_tokens.size() &&
             instruction.operand_tokens[pos].kind !=
                 cudasim::ptx::TokenKind::RBracket) {
        if (instruction.operand_tokens[pos].kind ==
                cudasim::ptx::TokenKind::Plus &&
            pos + 1 < instruction.operand_tokens.size()) {
          ++pos;
          if (instruction.operand_tokens[pos].kind ==
              cudasim::ptx::TokenKind::Number) {
            byte_offset += static_cast<std::size_t>(
                std::stoull(instruction.operand_tokens[pos].text));
          }
        }
        ++pos;
      }
    }

    if (!has_address || byte_offset + sizeof(float) > shared_memory.size()) {
      return false;
    }
    float fv = 0.0F;
    std::memcpy(&fv, shared_memory.data() + byte_offset, sizeof(fv));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &fv, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "atom.add.global.f32" || opcode == "atom.global.add.f32" ||
      opcode == "atom.add.global.u32") {
    std::size_t pos = 0;
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    auto address = parse_register(instruction.operand_tokens, pos);
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::Percent) {
      ++pos;
    }
    auto value = parse_register(instruction.operand_tokens, pos);
    if (!address || !value) {
      return false;
    }

    if (opcode == "atom.add.global.f32" || opcode == "atom.global.add.f32") {
      float fv = 0.0F;
      std::memcpy(&fv, &registers.values[value->index][lane], sizeof(fv));
      cudasim::mem::atomic_add_f32(
          reinterpret_cast<float *>(registers.values[address->index][lane]),
          fv);
    } else {
      cudasim::mem::atomic_add_u32(
          reinterpret_cast<std::uint32_t *>(
              registers.values[address->index][lane]),
          static_cast<std::uint32_t>(registers.values[value->index][lane]));
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mov.u64" || opcode == "mov.b64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    bool negative = false;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Minus) {
      negative = true;
      ++pos;
    }
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      const std::uint64_t value = static_cast<std::uint64_t>(
          std::stoull(instruction.operand_tokens[pos].text));
      registers.values[dst->index][lane] =
          negative
              ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(value))
              : value;
      next_pc = pc + 1;
      return true;
    }
    auto src = parse_register(instruction.operand_tokens, pos);
    if (!src) {
      return false;
    }
    registers.values[dst->index][lane] = registers.values[src->index][lane];
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "mov.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    auto value =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!value) {
      return false;
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &*value, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "neg.s32" || opcode == "not.b32" || opcode == "abs.s32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a) {
      return false;
    }
    std::uint32_t result = 0;
    if (opcode == "neg.s32") {
      result = static_cast<std::uint32_t>(-static_cast<std::int32_t>(*a));
    } else if (opcode == "not.b32") {
      result = ~*a;
    } else {
      const std::int32_t av = static_cast<std::int32_t>(*a);
      result = static_cast<std::uint32_t>(av < 0 ? -av : av);
    }
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "add.s32" || opcode == "sub.s32" || opcode == "and.b32" ||
      opcode == "or.b32" || opcode == "xor.b32" || opcode == "mul.lo.s32" ||
      opcode == "div.s32" || opcode == "rem.s32" || opcode == "max.s32" ||
      opcode == "min.s32" || opcode == "min.u32" || opcode == "max.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a || !b) {
      return false;
    }
    std::uint32_t result = 0;
    if (opcode == "add.s32") {
      result = *a + *b;
    } else if (opcode == "sub.s32") {
      result = *a - *b;
    } else if (opcode == "and.b32") {
      result = *a & *b;
    } else if (opcode == "or.b32") {
      result = *a | *b;
    } else if (opcode == "xor.b32") {
      result = *a ^ *b;
    } else if (opcode == "div.s32") {
      const auto av = static_cast<std::int32_t>(*a);
      const auto bv = static_cast<std::int32_t>(*b);
      result = bv == 0 ? 0U : static_cast<std::uint32_t>(av / bv);
    } else if (opcode == "rem.s32") {
      const auto av = static_cast<std::int32_t>(*a);
      const auto bv = static_cast<std::int32_t>(*b);
      result = bv == 0 ? 0U : static_cast<std::uint32_t>(av % bv);
    } else if (opcode == "max.s32") {
      result = static_cast<std::uint32_t>(std::max(
          static_cast<std::int32_t>(*a), static_cast<std::int32_t>(*b)));
    } else if (opcode == "min.s32") {
      result = static_cast<std::uint32_t>(std::min(
          static_cast<std::int32_t>(*a), static_cast<std::int32_t>(*b)));
    } else if (opcode == "min.u32") {
      result = std::min(*a, *b);
    } else if (opcode == "max.u32") {
      result = std::max(*a, *b);
    } else {
      result = *a * *b;
    }
    registers.values[dst->index][lane] = result;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "setp.eq.s64" || opcode == "setp.ne.s64" ||
      opcode == "setp.lt.s64" || opcode == "setp.le.s64" ||
      opcode == "setp.gt.s64" || opcode == "setp.ge.s64" ||
      opcode == "setp.eq.b64" || opcode == "setp.ne.b64") {
    std::size_t pos = 0;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    auto a = parse_register(instruction.operand_tokens, pos);
    if (!a || pred_index >= registers.predicates.size()) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind ==
               cudasim::ptx::TokenKind::Comma) {
      ++pos;
    }
    std::uint64_t bv = 0;
    bool negative = false;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Minus) {
      negative = true;
      ++pos;
    }
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      bv = static_cast<std::uint64_t>(
          std::stoll(instruction.operand_tokens[pos].text));
      if (negative) {
        bv = static_cast<std::uint64_t>(-static_cast<std::int64_t>(bv));
      }
    } else {
      auto b = parse_register(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      bv = registers.values[b->index][lane];
    }
    const std::uint64_t av = registers.values[a->index][lane];
    const auto av_signed = static_cast<std::int64_t>(av);
    const auto bv_signed = static_cast<std::int64_t>(bv);
    if (opcode == "setp.eq.s64" || opcode == "setp.eq.b64") {
      registers.predicates[pred_index][lane] = av == bv;
    } else if (opcode == "setp.ne.s64" || opcode == "setp.ne.b64") {
      registers.predicates[pred_index][lane] = av != bv;
    } else if (opcode == "setp.lt.s64") {
      registers.predicates[pred_index][lane] = av_signed < bv_signed;
    } else if (opcode == "setp.le.s64") {
      registers.predicates[pred_index][lane] = av_signed <= bv_signed;
    } else if (opcode == "setp.gt.s64") {
      registers.predicates[pred_index][lane] = av_signed > bv_signed;
    } else {
      registers.predicates[pred_index][lane] = av_signed >= bv_signed;
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "setp.lt.s32" || opcode == "setp.lt.u32" ||
      opcode == "setp.eq.s32" || opcode == "setp.gt.s32" ||
      opcode == "setp.le.s32" || opcode == "setp.eq.b32" ||
      opcode == "setp.gt.u32") {
    std::size_t pos = 0;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind !=
            cudasim::ptx::TokenKind::Percent) {
      return false;
    }
    ++pos;
    const std::string pred_name = instruction.operand_tokens[pos++].text;
    std::size_t pred_index = 0;
    for (std::size_t i = 1; i < pred_name.size(); ++i) {
      pred_index =
          pred_index * 10 + static_cast<std::size_t>(pred_name[i] - '0');
    }
    auto a =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    auto b =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!a || !b || pred_index >= registers.predicates.size()) {
      return false;
    }
    if (opcode == "setp.eq.s32" || opcode == "setp.eq.b32") {
      registers.predicates[pred_index][lane] = *a == *b;
    } else if (opcode == "setp.lt.s32") {
      registers.predicates[pred_index][lane] =
          static_cast<std::int32_t>(*a) < static_cast<std::int32_t>(*b);
    } else if (opcode == "setp.gt.s32") {
      registers.predicates[pred_index][lane] =
          static_cast<std::int32_t>(*a) > static_cast<std::int32_t>(*b);
    } else if (opcode == "setp.le.s32") {
      registers.predicates[pred_index][lane] =
          static_cast<std::int32_t>(*a) <= static_cast<std::int32_t>(*b);
    } else if (opcode == "setp.gt.u32") {
      registers.predicates[pred_index][lane] = *a > *b;
    } else {
      registers.predicates[pred_index][lane] = *a < *b;
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "cvt.rn.f32.s32" || opcode == "cvt.rn.f32.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !src) {
      return false;
    }
    const float value =
        opcode == "cvt.rn.f32.s32"
            ? static_cast<float>(static_cast<std::int32_t>(*src))
            : static_cast<float>(*src);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "cvt.rzi.s32.f32" || opcode == "cvt.rzi.u32.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a) {
      return false;
    }
    if (opcode == "cvt.rzi.s32.f32") {
      registers.values[dst->index][lane] =
          static_cast<std::uint32_t>(static_cast<std::int32_t>(*a));
    } else {
      registers.values[dst->index][lane] = static_cast<std::uint32_t>(*a);
    }
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "cvt.s64.s32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !src) {
      return false;
    }
    registers.values[dst->index][lane] =
        static_cast<std::uint64_t>(static_cast<std::int32_t>(*src));
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "cvt.u64.u32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src =
        parse_u32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !src) {
      return false;
    }
    registers.values[dst->index][lane] = static_cast<std::uint64_t>(*src);
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "cvt.u32.u64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src = parse_register(instruction.operand_tokens, pos);
    if (!dst || !src) {
      return false;
    }
    registers.values[dst->index][lane] =
        static_cast<std::uint32_t>(registers.values[src->index][lane]);
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "neg.f32" || opcode == "cvt.sat.f32.f32" ||
      opcode == "ex2.approx.ftz.f32" || opcode == "ex2.approx.f32" ||
      opcode == "abs.f32" || opcode == "abs.ftz.f32" ||
      opcode == "cvt.rmi.f32.f32" || opcode == "cvt.rpi.f32.f32" ||
      opcode == "cvt.rni.f32.f32" || opcode == "sin.approx.f32" ||
      opcode == "cos.approx.f32" || opcode == "lg2.approx.f32" ||
      opcode == "lg2.approx.ftz.f32" || opcode == "rcp.approx.ftz.f32" ||
      opcode == "rcp.rn.f32" || opcode == "cvt.rzi.f32.f32" ||
      opcode == "rsqrt.approx.f32" || opcode == "rsqrt.approx.ftz.f32" ||
      opcode == "sqrt.approx.f32" || opcode == "sqrt.approx.ftz.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    if (!dst || !a) {
      return false;
    }
    float result = 0.0F;
    if (opcode == "neg.f32") {
      result = -*a;
    } else if (opcode == "cvt.sat.f32.f32") {
      result = std::clamp(*a, 0.0F, 1.0F);
    } else if (opcode == "abs.f32" || opcode == "abs.ftz.f32") {
      result = std::fabs(*a);
    } else if (opcode == "cvt.rmi.f32.f32") {
      result = std::floor(*a);
    } else if (opcode == "cvt.rpi.f32.f32") {
      result = std::ceil(*a);
    } else if (opcode == "cvt.rni.f32.f32") {
      result = std::nearbyint(*a);
    } else if (opcode == "sin.approx.f32") {
      result = std::sin(*a);
    } else if (opcode == "cos.approx.f32") {
      result = std::cos(*a);
    } else if (opcode == "lg2.approx.f32" || opcode == "lg2.approx.ftz.f32") {
      result = std::log2(*a);
    } else if (opcode == "rcp.approx.ftz.f32" || opcode == "rcp.rn.f32") {
      result = 1.0F / *a;
    } else if (opcode == "cvt.rzi.f32.f32") {
      result = std::trunc(*a);
    } else if (opcode == "rsqrt.approx.f32" ||
               opcode == "rsqrt.approx.ftz.f32") {
      result = 1.0F / std::sqrt(*a);
    } else if (opcode == "sqrt.approx.f32" || opcode == "sqrt.approx.ftz.f32") {
      result = std::sqrt(*a);
    } else {
      result = std::exp2(*a);
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "div.rn.f32" || opcode == "div.approx.f32" ||
      opcode == "div.approx.ftz.f32" || opcode == "div.full.f32" ||
      opcode == "div.full.ftz.f32" || opcode == "div.rn.ftz.f32" ||
      opcode == "sub.f32" || opcode == "sqrt.rn.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a =
        parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    std::optional<float> b;
    if (opcode != "sqrt.rn.f32") {
      b = parse_f32_operand(instruction.operand_tokens, pos, registers, lane);
    }
    if (!dst || !a || (opcode != "sqrt.rn.f32" && !b)) {
      return false;
    }
    float result = 0.0F;
    if (opcode == "div.rn.f32" || opcode == "div.approx.f32" ||
        opcode == "div.approx.ftz.f32" || opcode == "div.full.f32" ||
        opcode == "div.full.ftz.f32" || opcode == "div.rn.ftz.f32") {
      result = *a / *b;
    } else if (opcode == "sub.f32") {
      result = *a - *b;
    } else {
      result = std::sqrt(*a);
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  next_pc = pc + 1;
  return true;
}

} // namespace

bool Executor::execute(const cudasim::ptx::FunctionIR &function,
                       const LaunchConfig &config,
                       const std::vector<std::uint8_t> &arguments,
                       const std::vector<cudasim::ptx::VariableIR> &variables) {
  std::unordered_map<std::string, std::size_t> shared_offsets;
  std::size_t shared_bytes = 0;
  for (const auto &variable : variables) {
    if (variable.space != "shared") {
      continue;
    }
    const std::size_t element_size = type_size(variable.type);
    const std::size_t count = variable.size == 0 ? 1 : variable.size;
    shared_offsets[variable.name] = shared_bytes;
    if (variable.space == "shared" && variable.size == 0) {
      shared_bytes += config.shared_memory;
    } else {
      shared_bytes += element_size * count;
    }
  }
  std::vector<std::uint8_t> shared_memory(shared_bytes, 0);

  std::unordered_map<std::string, std::size_t> parameter_offsets;
  std::size_t offset = 0;
  for (const auto &parameter : function.parameters) {
    parameter_offsets[parameter.name] = offset;
    offset += parameter.size;
  }

  std::vector<FlatInstruction> flat;
  std::unordered_map<std::string, std::size_t> labels;
  for (const auto &block : function.blocks) {
    if (!block.label.empty()) {
      labels[block.label] = flat.size();
    }
    for (const auto &instruction : block.instructions) {
      FlatInstruction decoded;
      decoded.opcode = instruction.opcode;
      decoded.predicate = instruction.predicate;
      decoded.operands = instruction.operands;
      decoded.operand_tokens = tokenize(instruction.operands);
      decoded.predicate_tokens = tokenize(instruction.predicate);
      flat.push_back(std::move(decoded));
    }
  }

  const unsigned int grid_x = config.grid[0];
  const unsigned int grid_y = config.grid[1];
  const unsigned int grid_z = config.grid[2];

  for (unsigned int gz = 0; gz < grid_z; ++gz) {
    for (unsigned int gy = 0; gy < grid_y; ++gy) {
      for (unsigned int gx = 0; gx < grid_x; ++gx) {
        BlockInfo block;
        block.index = {gx, gy, gz};
        block.dim = config.block;

        const std::size_t thread_count =
            static_cast<std::size_t>(config.block[0]) * config.block[1] *
            config.block[2];
        const std::size_t warp_count = (thread_count + 31) / 32;

        struct WarpRuntime {
          RegisterFile registers;
          std::vector<bool> active;
          std::vector<bool> blocked;
          std::vector<std::size_t> pc;
          bool finished = false;
        };

        std::vector<WarpRuntime> warps;
        warps.reserve(warp_count);
        for (std::size_t warp = 0; warp < warp_count; ++warp) {
          WarpRuntime runtime;
          runtime.active.assign(32, false);
          runtime.blocked.assign(32, false);
          runtime.pc.assign(32, 0);
          const std::size_t lane_base = warp * 32;
          for (std::size_t lane = 0; lane < 32; ++lane) {
            if (lane_base + lane < thread_count) {
              runtime.active[lane] = true;
            }
          }
          warps.push_back(std::move(runtime));
        }

        bool block_running = true;
        int arrived = 0;
        while (block_running) {
          bool any_unfinished = false;
          std::size_t active_threads = 0;
          for (const WarpRuntime &runtime : warps) {
            if (runtime.finished) {
              continue;
            }
            for (std::size_t lane = 0; lane < 32; ++lane) {
              if (runtime.active[lane]) {
                ++active_threads;
              }
            }
          }

          for (std::size_t warp = 0; warp < warps.size(); ++warp) {
            WarpRuntime &runtime = warps[warp];
            if (runtime.finished) {
              continue;
            }

            bool has_active = false;
            bool has_blocked = false;
            const std::size_t lane_base = warp * 32;

            std::vector<std::size_t> active_lanes;
            for (std::size_t lane = 0; lane < 32; ++lane) {
              if (runtime.blocked[lane]) {
                has_blocked = true;
              }
              if (runtime.active[lane] && !runtime.blocked[lane]) {
                active_lanes.push_back(lane);
              }
            }

            if (!active_lanes.empty()) {
              const std::size_t common_pc = runtime.pc[active_lanes.front()];
              bool same_pc = true;
              for (const std::size_t lane : active_lanes) {
                same_pc = same_pc && runtime.pc[lane] == common_pc;
              }
              if (same_pc && common_pc < flat.size() &&
                  (flat[common_pc].opcode == "ld.global.f32" ||
                   flat[common_pc].opcode == "st.global.f32")) {
                const bool store = flat[common_pc].opcode == "st.global.f32";
                if (coalesced_global_f32(flat[common_pc], runtime.registers,
                                         active_lanes, store)) {
                  for (const std::size_t lane : active_lanes) {
                    ++runtime.pc[lane];
                  }
                  any_unfinished = true;
                  continue;
                }
              }
            }

            for (std::size_t lane = 0; lane < 32; ++lane) {
              if (!runtime.active[lane] || runtime.blocked[lane]) {
                continue;
              }
              has_active = true;
              if (runtime.pc[lane] >= flat.size()) {
                runtime.active[lane] = false;
                continue;
              }

              const std::size_t current_pc = runtime.pc[lane];
              if (flat[current_pc].opcode == "bar.sync") {
                runtime.blocked[lane] = true;
                ++arrived;
                has_blocked = true;
                continue;
              }

              std::size_t next_pc = current_pc + 1;
              const bool still_active = execute_instruction(
                  flat[current_pc], config, block, runtime.registers, lane,
                  lane_base + lane, arguments, parameter_offsets, shared_memory,
                  shared_offsets, next_pc, current_pc, labels);
              runtime.pc[lane] = next_pc;
              if (!still_active) {
                runtime.active[lane] = false;
              }
            }

            if (has_blocked) {
              any_unfinished = true;
            } else if (has_active) {
              any_unfinished = true;
            } else {
              runtime.finished = true;
            }
          }

          if (arrived > 0 && arrived == static_cast<int>(active_threads)) {
            for (WarpRuntime &runtime : warps) {
              for (std::size_t lane = 0; lane < 32; ++lane) {
                if (runtime.blocked[lane]) {
                  ++runtime.pc[lane];
                  runtime.blocked[lane] = false;
                }
              }
            }
            arrived = 0;
          }

          if (arrived > 0) {
            any_unfinished = true;
          }

          if (!any_unfinished && arrived == 0) {
            block_running = false;
          }
        }
      }
    }
  }

  return true;
}

} // namespace cudasim::exec
