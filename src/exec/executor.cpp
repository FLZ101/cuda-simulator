#include "exec/executor.h"

#include "exec/block.h"
#include "exec/register_file.h"
#include "mem/atomics.h"
#include "mem/coalescer.h"
#include "ptx/lexer.h"
#include "ptx/token.h"

#include <algorithm>
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
  if (bank == 'd') {
    base = 64;
  } else if (bank == 'f') {
    base = 128;
  }
  ref.index = base + index;
  ref.size = (bank == 'd' || bank == 'f') ? 8 : 4;
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
  if (name == "%tid.x") {
    return static_cast<std::uint32_t>(lane_in_block);
  }
  if (name == "%ntid.x") {
    return block_dim_x;
  }
  if (name == "%ctaid.x") {
    return block_id_x;
  }
  if (name == "%nctaid.x") {
    return grid_dim_x;
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
    return true;
  }
  const bool value = registers.predicates[index][lane];
  return negate ? !value : value;
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

    std::vector<const void *> addresses;
    addresses.reserve(lanes.size());
    for (const std::size_t lane : lanes) {
      addresses.push_back(reinterpret_cast<const void *>(
          registers.values[address->index][lane]));
    }

    std::vector<std::uint32_t> values;
    cudasim::mem::coalesced_read_u32_pointers(addresses, values);
    for (std::size_t i = 0; i < lanes.size(); ++i) {
      registers.values[dst->index][lanes[i]] = values[i];
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
  while (pos < instruction.operand_tokens.size() &&
         instruction.operand_tokens[pos].kind !=
             cudasim::ptx::TokenKind::Percent) {
    ++pos;
  }
  auto value = parse_register(instruction.operand_tokens, pos);
  if (!address || !value) {
    return false;
  }

  std::vector<void *> addresses;
  std::vector<std::uint32_t> values;
  addresses.reserve(lanes.size());
  values.reserve(lanes.size());
  for (const std::size_t lane : lanes) {
    addresses.push_back(
        reinterpret_cast<void *>(registers.values[address->index][lane]));
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

  if (opcode == "ret" || opcode == "exit") {
    return false;
  }

  if (opcode == "bra") {
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

  if (opcode == "ld.param.u64" || opcode == "ld.param.u32" ||
      opcode == "ld.param.f32") {
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

  if (opcode == "mov.u32") {
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
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    auto c = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c) {
      return false;
    }
    const auto av = static_cast<std::int32_t>(registers.values[a->index][lane]);
    const auto bv = static_cast<std::int32_t>(registers.values[b->index][lane]);
    const auto cv = static_cast<std::int32_t>(registers.values[c->index][lane]);
    const auto result = static_cast<std::uint32_t>(av * bv + cv);
    registers.values[dst->index][lane] = result;
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

  if (opcode == "mul.wide.s32") {
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
    std::int64_t av =
        static_cast<std::int32_t>(registers.values[a->index][lane]);
    std::int64_t bv = 0;
    if (instruction.operand_tokens[pos].kind ==
        cudasim::ptx::TokenKind::Number) {
      bv = std::stoll(instruction.operand_tokens[pos].text);
    } else {
      auto b = parse_register(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      bv = static_cast<std::int32_t>(registers.values[b->index][lane]);
    }
    registers.values[dst->index][lane] = static_cast<std::uint64_t>(av * bv);
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "add.s64") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a) {
      return false;
    }
    registers.values[dst->index][lane] =
        registers.values[a->index][lane] + registers.values[b->index][lane];
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
    const auto *ptr =
        reinterpret_cast<const float *>(registers.values[address->index][lane]);
    std::uint32_t bits = 0;
    std::memcpy(&bits, ptr, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "add.f32") {
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
      if (immediate == "0f00000000") {
        bv = 0.0F;
      } else {
        bv = std::stof(immediate);
      }
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

  if (opcode == "mul.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    float av = 0.0F;
    float bv = 0.0F;
    std::memcpy(&av, &registers.values[a->index][lane], sizeof(av));
    std::memcpy(&bv, &registers.values[b->index][lane], sizeof(bv));
    const float result = av * bv;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    registers.values[dst->index][lane] = bits;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "fma.rn.f32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    auto c = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c) {
      return false;
    }
    float av = 0.0F;
    float bv = 0.0F;
    float cv = 0.0F;
    std::memcpy(&av, &registers.values[a->index][lane], sizeof(av));
    std::memcpy(&bv, &registers.values[b->index][lane], sizeof(bv));
    std::memcpy(&cv, &registers.values[c->index][lane], sizeof(cv));
    const float result = av * bv + cv;
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
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind !=
               cudasim::ptx::TokenKind::Percent) {
      ++pos;
    }
    auto value = parse_register(instruction.operand_tokens, pos);
    if (!address || !value) {
      return false;
    }
    auto *ptr =
        reinterpret_cast<float *>(registers.values[address->index][lane]);
    float fv = 0.0F;
    std::memcpy(&fv, &registers.values[value->index][lane], sizeof(fv));
    *ptr = fv;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "shl.b32") {
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
    registers.values[dst->index][lane] =
        static_cast<std::uint32_t>(registers.values[src->index][lane])
        << amount;
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "add.s32") {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_register(instruction.operand_tokens, pos);
    auto b = parse_register(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    const auto av = static_cast<std::int32_t>(registers.values[a->index][lane]);
    const auto bv = static_cast<std::int32_t>(registers.values[b->index][lane]);
    registers.values[dst->index][lane] = static_cast<std::uint32_t>(av + bv);
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "bar.sync") {
    next_pc = pc + 1;
    return true;
  }

  if (opcode == "setp.ne.s32") {
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
    std::uint32_t bv = 0;
    if (pos < instruction.operand_tokens.size() &&
        instruction.operand_tokens[pos].kind ==
            cudasim::ptx::TokenKind::Number) {
      bv = static_cast<std::uint32_t>(
          std::stoul(instruction.operand_tokens[pos].text));
    } else {
      auto b = parse_register(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      bv = static_cast<std::uint32_t>(registers.values[b->index][lane]);
    }
    registers.predicates[pred_index][lane] =
        static_cast<std::uint32_t>(registers.values[a->index][lane]) != bv;
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

  if (opcode == "atom.add.global.f32" || opcode == "atom.add.global.u32") {
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

    if (opcode == "atom.add.global.f32") {
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
    shared_bytes += element_size * count;
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
        while (block_running) {
          int arrived = 0;
          bool any_unfinished = false;

          for (std::size_t warp = 0; warp < warps.size(); ++warp) {
            WarpRuntime &runtime = warps[warp];
            if (runtime.finished) {
              continue;
            }

            bool warp_arrived = false;
            bool has_active = false;
            const std::size_t lane_base = warp * 32;

            std::vector<std::size_t> active_lanes;
            for (std::size_t lane = 0; lane < 32; ++lane) {
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
                warp_arrived = true;
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

            if (warp_arrived) {
              ++arrived;
              any_unfinished = true;
            } else if (has_active) {
              any_unfinished = true;
            } else {
              runtime.finished = true;
            }
          }

          if (arrived > 0 && arrived == static_cast<int>(warps.size())) {
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
