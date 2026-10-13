#include "jit/ptx_to_cpp.h"

#include "exec/register_file.h"
#include "ptx/lexer.h"
#include "ptx/token.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudasim::jit {

namespace {

using cudasim::ptx::IrInstruction;
using cudasim::ptx::Token;
using cudasim::ptx::TokenKind;

struct FlatInstruction {
  std::string opcode;
  std::string predicate;
  std::string operands;
  std::vector<Token> operand_tokens;
  std::vector<Token> predicate_tokens;
};

std::vector<Token> tokenize(const std::string &text) {
  cudasim::ptx::Lexer lexer(text);
  std::vector<Token> tokens;
  for (;;) {
    Token token = lexer.next();
    if (token.kind == TokenKind::End) {
      break;
    }
    tokens.push_back(std::move(token));
  }
  return tokens;
}

struct RegisterRef {
  std::size_t index = 0;
};

std::optional<RegisterRef> parse_register(const std::vector<Token> &tokens,
                                          std::size_t &pos) {
  while (pos < tokens.size() && tokens[pos].kind == TokenKind::Comma) {
    ++pos;
  }
  if (pos < tokens.size() && tokens[pos].kind == TokenKind::LBrace) {
    ++pos;
  }
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Percent) {
    return std::nullopt;
  }
  ++pos;
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Identifier) {
    return std::nullopt;
  }

  const std::string name = tokens[pos].text;
  ++pos;

  char bank = 'r';
  std::size_t number_start = 0;
  if (name.size() >= 2 && name[0] == 'r' && name[1] == 'd') {
    bank = 'd';
    number_start = 2;
  } else if (name.size() >= 2 && name[0] == 'r' && name[1] == 's') {
    bank = 's';
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
    if (name[i] < '0' || name[i] > '9') {
      return std::nullopt;
    }
    index = index * 10 + static_cast<std::size_t>(name[i] - '0');
  }

  std::size_t base = 0;
  if (bank == 'd') {
    base = cudasim::exec::kRegisterDBankBase;
  } else if (bank == 'f') {
    base = cudasim::exec::kRegisterFBankBase;
  } else if (bank == 's') {
    base = cudasim::exec::kRegisterSBankBase;
  }
  if (index >= cudasim::exec::kRegisterBankSize) {
    return std::nullopt;
  }

  if (pos < tokens.size() && tokens[pos].kind == TokenKind::RBrace) {
    ++pos;
  }
  return RegisterRef{base + index};
}

bool is_special_base(const std::string &name) {
  return name == "tid" || name == "ntid" || name == "ctaid" ||
         name == "nctaid" || name == "laneid" || name == "warpid";
}

std::optional<std::string> parse_special(const std::vector<Token> &tokens,
                                         std::size_t &pos) {
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Percent) {
    return std::nullopt;
  }
  const std::size_t start = pos;
  ++pos;
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Identifier ||
      !is_special_base(tokens[pos].text)) {
    pos = start;
    return std::nullopt;
  }

  const std::string base = tokens[pos].text;
  ++pos;
  if ((base == "tid" || base == "ntid" || base == "ctaid" ||
       base == "nctaid")) {
    if (pos + 1 >= tokens.size() || tokens[pos].kind != TokenKind::Dot ||
        tokens[pos + 1].kind != TokenKind::Identifier) {
      pos = start;
      return std::nullopt;
    }
    const std::string coordinate = tokens[pos + 1].text;
    if (coordinate != "x" && coordinate != "y" && coordinate != "z") {
      pos = start;
      return std::nullopt;
    }
    pos += 2;
    return "%" + base + "." + coordinate;
  }
  return "%" + base;
}

std::string special_expression(const std::string &name) {
  if (name == "%tid.x") {
    return "tid_x";
  }
  if (name == "%tid.y") {
    return "tid_y";
  }
  if (name == "%tid.z") {
    return "tid_z";
  }
  if (name == "%ntid.x") {
    return "ntid_x";
  }
  if (name == "%ntid.y") {
    return "ntid_y";
  }
  if (name == "%ntid.z") {
    return "ntid_z";
  }
  if (name == "%ctaid.x") {
    return "ctaid_x";
  }
  if (name == "%ctaid.y") {
    return "ctaid_y";
  }
  if (name == "%ctaid.z") {
    return "ctaid_z";
  }
  if (name == "%nctaid.x") {
    return "nctaid_x";
  }
  if (name == "%nctaid.y") {
    return "nctaid_y";
  }
  if (name == "%nctaid.z") {
    return "nctaid_z";
  }
  if (name == "%laneid") {
    return "laneid";
  }
  return "warpid";
}

struct Operand {
  enum class Kind { Register, Immediate, Special };
  Kind kind = Kind::Immediate;
  std::size_t index = 0;
  std::uint64_t immediate = 0;
  std::string special;
};

std::optional<Operand> parse_integer_operand(const std::vector<Token> &tokens,
                                             std::size_t &pos) {
  while (pos < tokens.size() && tokens[pos].kind == TokenKind::Comma) {
    ++pos;
  }
  if (pos >= tokens.size()) {
    return std::nullopt;
  }

  bool negative = false;
  if (tokens[pos].kind == TokenKind::Minus) {
    negative = true;
    ++pos;
  }

  if (pos < tokens.size() && tokens[pos].kind == TokenKind::Number) {
    const std::string text = tokens[pos].text;
    const std::uint64_t value =
        text.size() == 10 && text[0] == '0' && text[1] == 'f'
            ? static_cast<std::uint64_t>(
                  std::stoul(text.substr(2), nullptr, 16))
            : static_cast<std::uint64_t>(std::stoull(text, nullptr, 0));
    ++pos;
    Operand result;
    result.kind = Operand::Kind::Immediate;
    result.immediate = negative ? static_cast<std::uint64_t>(0) - value : value;
    return result;
  }

  if (auto special = parse_special(tokens, pos)) {
    Operand result;
    result.kind = Operand::Kind::Special;
    result.special = *special;
    return result;
  }

  auto reg = parse_register(tokens, pos);
  if (!reg) {
    return std::nullopt;
  }
  Operand result;
  result.kind = Operand::Kind::Register;
  result.index = reg->index;
  return result;
}

struct FloatOperand {
  enum class Kind { Register, Immediate, Special };
  Kind kind = Kind::Immediate;
  std::size_t index = 0;
  std::uint32_t bits = 0;
  std::string special;
};

std::optional<FloatOperand>
parse_float_operand(const std::vector<Token> &tokens, std::size_t &pos) {
  while (pos < tokens.size() && tokens[pos].kind == TokenKind::Comma) {
    ++pos;
  }
  if (pos >= tokens.size()) {
    return std::nullopt;
  }

  if (tokens[pos].kind == TokenKind::Number) {
    const std::string text = tokens[pos].text;
    ++pos;
    FloatOperand result;
    result.kind = FloatOperand::Kind::Immediate;
    if (text.size() == 10 && text[0] == '0' && text[1] == 'f') {
      result.bits =
          static_cast<std::uint32_t>(std::stoul(text.substr(2), nullptr, 16));
    } else {
      const float value = std::stof(text);
      std::memcpy(&result.bits, &value, sizeof(result.bits));
    }
    return result;
  }

  if (auto special = parse_special(tokens, pos)) {
    FloatOperand result;
    result.kind = FloatOperand::Kind::Special;
    result.special = *special;
    return result;
  }

  auto reg = parse_register(tokens, pos);
  if (!reg) {
    return std::nullopt;
  }
  FloatOperand result;
  result.kind = FloatOperand::Kind::Register;
  result.index = reg->index;
  return result;
}

std::string base_expression(const Operand &operand) {
  if (operand.kind == Operand::Kind::Register) {
    return "R[" + std::to_string(operand.index) + "][lane]";
  }
  if (operand.kind == Operand::Kind::Special) {
    return "static_cast<uint64_t>(" + special_expression(operand.special) + ")";
  }
  return "static_cast<uint64_t>(" + std::to_string(operand.immediate) + "ULL)";
}

std::string u32_expression(const Operand &operand) {
  return "static_cast<uint32_t>(" + base_expression(operand) + ")";
}

std::string i32_expression(const Operand &operand) {
  return "static_cast<int32_t>(" + base_expression(operand) + ")";
}

std::string u64_expression(const Operand &operand) {
  return "static_cast<uint64_t>(" + base_expression(operand) + ")";
}

std::string i64_expression(const Operand &operand) {
  return "static_cast<int64_t>(" + base_expression(operand) + ")";
}

std::string f32_expression(const FloatOperand &operand) {
  if (operand.kind == FloatOperand::Kind::Register) {
    return "bits_to_f32(static_cast<uint32_t>(R[" +
           std::to_string(operand.index) + "][lane]))";
  }
  if (operand.kind == FloatOperand::Kind::Special) {
    return "static_cast<float>(" + special_expression(operand.special) + ")";
  }
  std::ostringstream bits;
  bits << "0x" << std::hex << operand.bits << "u";
  return "bits_to_f32(" + bits.str() + ")";
}

std::optional<std::size_t>
parse_predicate_index(const std::vector<Token> &tokens, std::size_t &pos) {
  while (pos < tokens.size() && tokens[pos].kind == TokenKind::Comma) {
    ++pos;
  }
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Percent) {
    return std::nullopt;
  }
  ++pos;
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Identifier ||
      tokens[pos].text.empty() || tokens[pos].text[0] != 'p') {
    return std::nullopt;
  }

  const std::string name = tokens[pos].text;
  ++pos;
  std::size_t index = 0;
  for (std::size_t i = 1; i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') {
      return std::nullopt;
    }
    index = index * 10 + static_cast<std::size_t>(name[i] - '0');
  }
  if (index >= 256) {
    return std::nullopt;
  }
  return index;
}

void update_register_limits(const std::vector<Token> &tokens,
                            std::size_t &max_register,
                            std::size_t &max_predicate) {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i].kind != TokenKind::Percent) {
      continue;
    }
    std::size_t register_pos = i;
    if (auto reg = parse_register(tokens, register_pos)) {
      max_register = std::max(max_register, reg->index);
      continue;
    }
    std::size_t predicate_pos = i;
    if (auto predicate = parse_predicate_index(tokens, predicate_pos)) {
      max_predicate = std::max(max_predicate, *predicate);
    }
  }
}

std::optional<std::string>
predicate_expression(const std::vector<Token> &tokens) {
  if (tokens.empty()) {
    return "true";
  }
  std::size_t pos = 0;
  bool negate = false;
  if (tokens[pos].kind == TokenKind::At) {
    ++pos;
  }
  if (pos < tokens.size() && tokens[pos].kind == TokenKind::Exclaim) {
    negate = true;
    ++pos;
  }
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Percent) {
    return std::nullopt;
  }
  const std::size_t percent = pos;
  ++pos;
  if (pos >= tokens.size() || tokens[pos].kind != TokenKind::Identifier) {
    return std::nullopt;
  }
  const std::string name = tokens[pos].text;
  if (name.empty() || name[0] != 'p') {
    return std::nullopt;
  }
  std::size_t index = 0;
  for (std::size_t i = 1; i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') {
      return std::nullopt;
    }
    index = index * 10 + static_cast<std::size_t>(name[i] - '0');
  }
  if (index >= 256) {
    return std::nullopt;
  }
  (void)percent;
  const std::string expr = "P[" + std::to_string(index) + "][lane]";
  return negate ? "(!" + expr + ")" : expr;
}

bool find_label(const std::unordered_map<std::string, std::size_t> &labels,
                const std::string &label, std::size_t &target) {
  const auto it = labels.find(label);
  if (it == labels.end()) {
    return false;
  }
  target = it->second;
  return true;
}

struct MemoryOperand {
  Operand address;
  std::int64_t immediate = 0;
};

std::optional<MemoryOperand>
parse_memory_operand(const std::vector<Token> &tokens, std::size_t &pos) {
  while (pos < tokens.size() && tokens[pos].kind != TokenKind::LBracket) {
    ++pos;
  }
  if (pos >= tokens.size()) {
    return std::nullopt;
  }
  ++pos;

  auto address = parse_integer_operand(tokens, pos);
  if (!address) {
    return std::nullopt;
  }

  std::int64_t immediate = 0;
  bool negative = false;
  if (pos < tokens.size() && tokens[pos].kind == TokenKind::Minus) {
    negative = true;
    ++pos;
  } else if (pos < tokens.size() && tokens[pos].kind == TokenKind::Plus) {
    ++pos;
    if (pos < tokens.size() && tokens[pos].kind == TokenKind::Minus) {
      negative = true;
      ++pos;
    }
  }
  if (pos < tokens.size() && tokens[pos].kind == TokenKind::Number) {
    immediate =
        static_cast<std::int64_t>(std::stoll(tokens[pos].text, nullptr, 0));
    if (negative) {
      immediate = -immediate;
    }
    ++pos;
  }

  return MemoryOperand{*address, immediate};
}

class Codegen {
public:
  explicit Codegen(const cudasim::ptx::FunctionIR &function)
      : function_(function) {}

  std::optional<std::string> generate() {
    for (const auto &parameter : function_.parameters) {
      parameter_offsets_[parameter.name] = next_parameter_offset_;
      next_parameter_offset_ += parameter.size;
    }

    std::size_t flat_index = 0;
    for (const auto &block : function_.blocks) {
      if (!block.label.empty()) {
        labels_[block.label] = flat_index;
      }
      for (const auto &instruction : block.instructions) {
        FlatInstruction decoded;
        decoded.opcode = instruction.opcode;
        decoded.predicate = instruction.predicate;
        decoded.operands = instruction.operands;
        decoded.operand_tokens = tokenize(instruction.operands);
        decoded.predicate_tokens = tokenize(instruction.predicate);
        flat_.push_back(std::move(decoded));
        ++flat_index;
      }
    }

    if (flat_.empty()) {
      return std::nullopt;
    }

    std::size_t max_register = 0;
    std::size_t max_predicate = 0;
    for (const auto &instruction : flat_) {
      update_register_limits(instruction.operand_tokens, max_register,
                             max_predicate);
      update_register_limits(instruction.predicate_tokens, max_register,
                             max_predicate);
    }
    const std::size_t register_count =
        std::max<std::size_t>(1, max_register + 1);
    const std::size_t predicate_count =
        std::max<std::size_t>(1, max_predicate + 1);

    out_ << "// native kernel: " << function_.name << "\n";
    out_ << preamble(register_count, predicate_count);
    out_ << "  const unsigned int bdx = block[0];\n";
    out_ << "  const unsigned int bdy = block[1];\n";
    out_ << "  const unsigned int bdz = block[2];\n";
    out_ << "  const unsigned int gdx = grid[0];\n";
    out_ << "  const unsigned int gdy = grid[1];\n";
    out_ << "  const unsigned int gdz = grid[2];\n";
    out_ << "  const unsigned int thread_count = bdx * bdy * bdz;\n";
    out_ << "  for (unsigned int warp = 0; warp < (thread_count + 31u) / 32u; "
            "++warp) {\n";
    out_ << "    std::array<std::array<uint64_t, 32>, " << register_count
         << "> R{};\n";
    out_ << "    std::array<std::array<uint8_t, 32>, " << predicate_count
         << "> P{};\n";
    out_ << "    for (unsigned int lane = 0; lane < 32u; ++lane) {\n";
    out_ << "      const unsigned int tid = warp * 32u + lane;\n";
    out_ << "      if (tid >= thread_count) continue;\n";
    out_ << "      const uint32_t lane_in_block = tid;\n";
    out_ << "      const uint32_t tid_x = lane_in_block % bdx;\n";
    out_ << "      const uint32_t tid_y = (lane_in_block / bdx) % std::max(1u, "
            "bdy);\n";
    out_ << "      const uint32_t tid_z = lane_in_block / (bdx * std::max(1u, "
            "bdy));\n";
    out_ << "      const uint32_t laneid = lane;\n";
    out_ << "      const uint32_t warpid = warp;\n";
    out_ << "      const uint32_t ntid_x = bdx, ntid_y = bdy, ntid_z = bdz;\n";
    out_ << "      const uint32_t ctaid_x = bx, ctaid_y = by, ctaid_z = bz;\n";
    out_ << "      const uint32_t nctaid_x = gdx, nctaid_y = gdy, nctaid_z = "
            "gdz;\n";
    out_ << "      int pc = 0;\n";
    out_ << "      bool active = true;\n";
    out_ << "      while (active) {\n";
    out_ << "        switch (pc) {\n";

    for (std::size_t i = 0; i < flat_.size(); ++i) {
      if (!emit_instruction(i, flat_[i])) {
        return std::nullopt;
      }
    }
    out_ << "          case " << flat_.size() << ": active = false; break;\n";
    out_ << "          default: active = false; break;\n";
    out_ << "        }\n";
    out_ << "      }\n";
    out_ << "    }\n";
    out_ << "  }\n";
    out_ << "  return 0;\n";
    out_ << "}\n";
    return out_.str();
  }

private:
  std::string preamble(std::size_t register_count,
                       std::size_t predicate_count) {
    (void)register_count;
    (void)predicate_count;
    std::ostringstream result;
    result << R"CPP(#include <algorithm>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

static inline float bits_to_f32(u32 bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline u64 f32_to_u64(float value) {
  u32 bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<u64>(bits);
}

static inline u64 load_arg_u64(const unsigned char *args, std::size_t offset) {
  u64 value = 0;
  std::memcpy(&value, args + offset, sizeof(value));
  return value;
}

static inline u32 load_arg_u32(const unsigned char *args, std::size_t offset) {
  u32 value = 0;
  std::memcpy(&value, args + offset, sizeof(value));
  return value;
}

static inline u64 load_arg_f32(const unsigned char *args, std::size_t offset) {
  u32 bits = load_arg_u32(args, offset);
  return static_cast<u64>(bits);
}

static inline u32 load_global_u32(u64 address) {
  u32 value = 0;
  std::memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
  return value;
}

static inline u64 load_global_u64(u64 address) {
  u64 value = 0;
  std::memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
  return value;
}

static inline void store_global_u32(u64 address, u32 value) {
  std::memcpy(reinterpret_cast<void *>(address), &value, sizeof(value));
}

static inline void store_global_u64(u64 address, u64 value) {
  std::memcpy(reinterpret_cast<void *>(address), &value, sizeof(value));
}

static inline u32 float_to_bits32(float value) {
  u32 bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline u32 bfe_u32(u32 source, u32 bit, u32 width) {
  if (width == 0) {
    return 0;
  }
  bit &= 31u;
  if (width >= 32u) {
    return source >> bit;
  }
  const u32 mask = (1u << width) - 1u;
  return (source >> bit) & mask;
}

static inline u32 bfe_s32(u32 source, u32 bit, u32 width) {
  if (width == 0) {
    return 0;
  }
  bit &= 31u;
  u32 result = 0;
  if (width >= 32u) {
    result = static_cast<u32>(static_cast<i32>(source) >> bit);
  } else {
    const u32 mask = (1u << width) - 1u;
    result = (source >> bit) & mask;
    if ((result & (1u << (width - 1u))) != 0u) {
      result |= ~mask;
    }
  }
  return result;
}

static inline u32 mul_hi_u32(u32 a, u32 b) {
  return static_cast<u32>((static_cast<u64>(a) * static_cast<u64>(b)) >> 32u);
}

extern "C" int cudasim_native_kernel(const unsigned char *args,
                                     const unsigned int *block,
                                     const unsigned int *grid,
                                     unsigned int bx, unsigned int by,
                                     unsigned int bz) {
)CPP";
    return result.str();
  }

  bool emit_predicated_body(std::size_t pc, const FlatInstruction &instruction,
                            const std::string &body) {
    auto predicate = predicate_expression(instruction.predicate_tokens);
    if (!predicate) {
      return false;
    }
    out_ << "          case " << pc << ": {\n";
    out_ << "            if (" << *predicate << ") {\n";
    out_ << body;
    out_ << "            }\n";
    out_ << "            pc = " << (pc + 1) << ";\n";
    out_ << "            break;\n";
    out_ << "          }\n";
    return true;
  }

  bool emit_branch(std::size_t pc, const FlatInstruction &instruction) {
    auto predicate = predicate_expression(instruction.predicate_tokens);
    if (!predicate || instruction.operand_tokens.empty() ||
        instruction.operand_tokens[0].kind != TokenKind::Dollar) {
      return false;
    }
    std::string label;
    if (instruction.operand_tokens.size() > 1 &&
        instruction.operand_tokens[1].kind == TokenKind::Identifier) {
      label = instruction.operand_tokens[1].text;
    } else {
      return false;
    }
    std::size_t target = 0;
    if (!find_label(labels_, label, target)) {
      return false;
    }
    out_ << "          case " << pc << ": {\n";
    out_ << "            if (" << *predicate << ") { pc = " << target
         << "; } else { pc = " << (pc + 1) << "; }\n";
    out_ << "            break;\n";
    out_ << "          }\n";
    return true;
  }

  bool emit_ret(std::size_t pc, const FlatInstruction &instruction) {
    auto predicate = predicate_expression(instruction.predicate_tokens);
    if (!predicate) {
      return false;
    }
    out_ << "          case " << pc << ": {\n";
    out_ << "            if (" << *predicate << ") active = false;\n";
    out_ << "            pc = " << (pc + 1) << ";\n";
    out_ << "            break;\n";
    out_ << "          }\n";
    return true;
  }

  bool emit_param_load(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind != TokenKind::LBracket) {
      ++pos;
    }
    if (pos >= instruction.operand_tokens.size()) {
      return false;
    }
    ++pos;
    if (pos >= instruction.operand_tokens.size() ||
        instruction.operand_tokens[pos].kind != TokenKind::Identifier) {
      return false;
    }
    const std::string parameter = instruction.operand_tokens[pos].text;
    const auto offset_it = parameter_offsets_.find(parameter);
    if (offset_it == parameter_offsets_.end()) {
      return false;
    }
    const std::size_t offset = offset_it->second;
    const std::string &opcode = instruction.opcode;
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    if (opcode == "ld.param.b64" || opcode == "ld.param.u64") {
      body << "R[" << dst->index << "][lane] = load_arg_u64(args, " << offset
           << ");\n";
    } else if (opcode == "ld.param.b32" || opcode == "ld.param.u32") {
      body << "R[" << dst->index << "][lane] = load_arg_u32(args, " << offset
           << ");\n";
    } else if (opcode == "ld.param.f32") {
      body << "R[" << dst->index << "][lane] = load_arg_f32(args, " << offset
           << ");\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_mov(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;

    if (opcode == "mov.f32") {
      auto src = parse_float_operand(instruction.operand_tokens, pos);
      if (!src) {
        return false;
      }
      body << "R[" << dst->index << "][lane] = f32_to_u64("
           << f32_expression(*src) << ");\n";
    } else if (opcode == "mov.u32" || opcode == "mov.b32" ||
               opcode == "mov.s32") {
      auto src = parse_integer_operand(instruction.operand_tokens, pos);
      if (!src) {
        return false;
      }
      body << "R[" << dst->index << "][lane] = static_cast<uint64_t>("
           << u32_expression(*src) << ");\n";
    } else if (opcode == "mov.u64" || opcode == "mov.b64" ||
               opcode == "mov.s64") {
      auto src = parse_integer_operand(instruction.operand_tokens, pos);
      if (!src) {
        return false;
      }
      body << "R[" << dst->index << "][lane] = " << u64_expression(*src)
           << ";\n";
    } else if (opcode == "mov.u16" || opcode == "mov.b16") {
      auto src = parse_integer_operand(instruction.operand_tokens, pos);
      if (!src) {
        return false;
      }
      body << "R[" << dst->index << "][lane] = (" << u32_expression(*src)
           << " & 0xFFFFu);\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_global_load(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    auto memory = parse_memory_operand(instruction.operand_tokens, pos);
    if (!memory) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    const std::string address =
        "(" + u64_expression(memory->address) +
        " + static_cast<uint64_t>(static_cast<int64_t>(" +
        std::to_string(memory->immediate) + ")))";
    if (instruction.opcode == "ld.global.b32" ||
        instruction.opcode == "ld.global.u32") {
      body << "R[" << dst->index << "][lane] = load_global_u32(" << address
           << ");\n";
    } else if (instruction.opcode == "ld.global.f32") {
      body << "R[" << dst->index << "][lane] = load_global_u32(" << address
           << ");\n";
    } else if (instruction.opcode == "ld.global.b64" ||
               instruction.opcode == "ld.global.u64") {
      body << "R[" << dst->index << "][lane] = load_global_u64(" << address
           << ");\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_global_store(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto memory = parse_memory_operand(instruction.operand_tokens, pos);
    if (!memory) {
      return false;
    }
    while (pos < instruction.operand_tokens.size() &&
           instruction.operand_tokens[pos].kind != TokenKind::Percent) {
      ++pos;
    }
    auto value = parse_integer_operand(instruction.operand_tokens, pos);
    if (!value) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    const std::string address =
        "(" + u64_expression(memory->address) +
        " + static_cast<uint64_t>(static_cast<int64_t>(" +
        std::to_string(memory->immediate) + ")))";
    if (instruction.opcode == "st.global.b32" ||
        instruction.opcode == "st.global.u32") {
      body << "store_global_u32(" << address << ", " << u32_expression(*value)
           << ");\n";
    } else if (instruction.opcode == "st.global.f32") {
      body << "store_global_u32(" << address << ", float_to_bits32("
           << f32_expression(FloatOperand{FloatOperand::Kind::Register,
                                          value->index, 0, value->special})
           << "));\n";
    } else if (instruction.opcode == "st.global.b64" ||
               instruction.opcode == "st.global.u64") {
      body << "store_global_u64(" << address << ", " << u64_expression(*value)
           << ");\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_setp_32(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto pred_index = parse_predicate_index(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!pred_index || !a || !b) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string comparison;
    if (opcode == "setp.eq.s32" || opcode == "setp.eq.b32" ||
        opcode == "setp.eq.u32") {
      comparison = u32_expression(*a) + " == " + u32_expression(*b);
    } else if (opcode == "setp.ne.s32" || opcode == "setp.ne.b32" ||
               opcode == "setp.ne.u32") {
      comparison = u32_expression(*a) + " != " + u32_expression(*b);
    } else if (opcode == "setp.lt.s32") {
      comparison = i32_expression(*a) + " < " + i32_expression(*b);
    } else if (opcode == "setp.lt.u32") {
      comparison = u32_expression(*a) + " < " + u32_expression(*b);
    } else if (opcode == "setp.le.s32") {
      comparison = i32_expression(*a) + " <= " + i32_expression(*b);
    } else if (opcode == "setp.le.u32") {
      comparison = u32_expression(*a) + " <= " + u32_expression(*b);
    } else if (opcode == "setp.gt.s32") {
      comparison = i32_expression(*a) + " > " + i32_expression(*b);
    } else if (opcode == "setp.gt.u32") {
      comparison = u32_expression(*a) + " > " + u32_expression(*b);
    } else if (opcode == "setp.ge.s32") {
      comparison = i32_expression(*a) + " >= " + i32_expression(*b);
    } else if (opcode == "setp.ge.u32") {
      comparison = u32_expression(*a) + " >= " + u32_expression(*b);
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "P[" << *pred_index << "][lane] = " << comparison << ";\n";
    return true;
  }

  bool emit_setp_64(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto pred_index = parse_predicate_index(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!pred_index || !a || !b) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string comparison;
    if (opcode == "setp.eq.s64" || opcode == "setp.eq.b64" ||
        opcode == "setp.eq.u64") {
      comparison = u64_expression(*a) + " == " + u64_expression(*b);
    } else if (opcode == "setp.ne.s64" || opcode == "setp.ne.b64" ||
               opcode == "setp.ne.u64") {
      comparison = u64_expression(*a) + " != " + u64_expression(*b);
    } else if (opcode == "setp.lt.s64") {
      comparison = i64_expression(*a) + " < " + i64_expression(*b);
    } else if (opcode == "setp.lt.u64") {
      comparison = u64_expression(*a) + " < " + u64_expression(*b);
    } else if (opcode == "setp.le.s64") {
      comparison = i64_expression(*a) + " <= " + i64_expression(*b);
    } else if (opcode == "setp.le.u64") {
      comparison = u64_expression(*a) + " <= " + u64_expression(*b);
    } else if (opcode == "setp.gt.s64") {
      comparison = i64_expression(*a) + " > " + i64_expression(*b);
    } else if (opcode == "setp.gt.u64") {
      comparison = u64_expression(*a) + " > " + u64_expression(*b);
    } else if (opcode == "setp.ge.s64") {
      comparison = i64_expression(*a) + " >= " + i64_expression(*b);
    } else if (opcode == "setp.ge.u64") {
      comparison = u64_expression(*a) + " >= " + u64_expression(*b);
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "P[" << *pred_index << "][lane] = " << comparison << ";\n";
    return true;
  }

  bool emit_integer_binary(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string expression;
    if (opcode == "add.s32" || opcode == "add.u32") {
      expression = u32_expression(*a) + " + " + u32_expression(*b);
    } else if (opcode == "sub.s32" || opcode == "sub.u32") {
      expression = u32_expression(*a) + " - " + u32_expression(*b);
    } else if (opcode == "and.b32") {
      expression = u32_expression(*a) + " & " + u32_expression(*b);
    } else if (opcode == "or.b32") {
      expression = u32_expression(*a) + " | " + u32_expression(*b);
    } else if (opcode == "xor.b32") {
      expression = u32_expression(*a) + " ^ " + u32_expression(*b);
    } else if (opcode == "mul.lo.s32" || opcode == "mul.lo.u32") {
      expression = u32_expression(*a) + " * " + u32_expression(*b);
    } else if (opcode == "max.s32") {
      expression = "static_cast<uint32_t>(std::max(" + i32_expression(*a) +
                   ", " + i32_expression(*b) + "))";
    } else if (opcode == "min.s32") {
      expression = "static_cast<uint32_t>(std::min(" + i32_expression(*a) +
                   ", " + i32_expression(*b) + "))";
    } else if (opcode == "max.u32") {
      expression =
          "std::max(" + u32_expression(*a) + ", " + u32_expression(*b) + ")";
    } else if (opcode == "min.u32") {
      expression =
          "std::min(" + u32_expression(*a) + ", " + u32_expression(*b) + ")";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = static_cast<uint64_t>("
         << expression << ");\n";
    return true;
  }

  bool emit_integer_unary(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a) {
      return false;
    }
    std::string expression;
    if (instruction.opcode == "neg.s32") {
      expression = "static_cast<uint32_t>(-" + i32_expression(*a) + ")";
    } else if (instruction.opcode == "not.b32") {
      expression = "~" + u32_expression(*a);
    } else if (instruction.opcode == "abs.s32") {
      expression =
          "static_cast<uint32_t>(std::abs(" + i32_expression(*a) + "))";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = static_cast<uint64_t>("
         << expression << ");\n";
    return true;
  }

  bool emit_integer_64(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string expression;
    if (opcode == "add.s64" || opcode == "add.u64") {
      expression = u64_expression(*a) + " + " + u64_expression(*b);
    } else if (opcode == "sub.s64" || opcode == "sub.u64") {
      expression = u64_expression(*a) + " - " + u64_expression(*b);
    } else if (opcode == "and.b64") {
      expression = u64_expression(*a) + " & " + u64_expression(*b);
    } else if (opcode == "or.b64") {
      expression = u64_expression(*a) + " | " + u64_expression(*b);
    } else if (opcode == "xor.b64") {
      expression = u64_expression(*a) + " ^ " + u64_expression(*b);
    } else if (opcode == "shl.b64") {
      expression = u64_expression(*a) + " << " + u64_expression(*b);
    } else if (opcode == "shr.u64") {
      expression = u64_expression(*a) + " >> " + u64_expression(*b);
    } else if (opcode == "shr.s64") {
      expression = "static_cast<uint64_t>(static_cast<int64_t>(" +
                   u64_expression(*a) + ") >> " + u64_expression(*b) + ")";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = " << expression << ";\n";
    return true;
  }

  bool emit_shift_32(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    std::string expression;
    if (instruction.opcode == "shl.b32") {
      expression = "(" + u32_expression(*a) + " << " + u32_expression(*b) + ")";
    } else if (instruction.opcode == "shr.u32") {
      expression = "(" + u32_expression(*a) + " >> " + u32_expression(*b) + ")";
    } else if (instruction.opcode == "shr.s32") {
      expression = "(static_cast<uint32_t>(static_cast<int32_t>(" +
                   u32_expression(*a) + ") >> " + u32_expression(*b) + "))";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = static_cast<uint64_t>("
         << expression << ");\n";
    return true;
  }

  bool emit_wide_mul(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    std::string expression;
    if (instruction.opcode == "mul.wide.s32") {
      expression = "static_cast<uint64_t>(static_cast<int64_t>(" +
                   i32_expression(*a) + ") * static_cast<int64_t>(" +
                   i32_expression(*b) + "))";
    } else if (instruction.opcode == "mul.wide.u32") {
      expression = "static_cast<uint64_t>(static_cast<uint64_t>(" +
                   u32_expression(*a) + ") * static_cast<uint64_t>(" +
                   u32_expression(*b) + "))";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = " << expression << ";\n";
    return true;
  }

  bool emit_mad_wide(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    auto c = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c) {
      return false;
    }
    std::string expression;
    if (instruction.opcode == "mad.wide.s32") {
      expression = "(static_cast<uint64_t>(static_cast<int64_t>(" +
                   i32_expression(*a) + ") * static_cast<int64_t>(" +
                   i32_expression(*b) + ") + static_cast<int64_t>(" +
                   u64_expression(*c) + ")))";
    } else if (instruction.opcode == "mad.wide.u32") {
      expression = "(static_cast<uint64_t>(static_cast<uint64_t>(" +
                   u32_expression(*a) + ") * static_cast<uint64_t>(" +
                   u32_expression(*b) + ") + " + u64_expression(*c) + "))";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = " << expression << ";\n";
    return true;
  }

  bool emit_mad_lo(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    auto c = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = static_cast<uint64_t>("
         << "static_cast<uint32_t>(static_cast<int32_t>(" << i32_expression(*a)
         << ") * static_cast<int32_t>(" << i32_expression(*b)
         << ") + static_cast<int32_t>(" << i32_expression(*c) << ")));\n";
    return true;
  }

  bool emit_convert(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !src) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string expression;
    if (opcode == "cvt.s64.s32") {
      expression = "static_cast<uint64_t>(static_cast<int64_t>(" +
                   i32_expression(*src) + "))";
    } else if (opcode == "cvt.u64.u32") {
      expression = "static_cast<uint64_t>(" + u32_expression(*src) + ")";
    } else if (opcode == "cvt.u32.u64") {
      expression = "static_cast<uint64_t>(static_cast<uint32_t>(" +
                   u64_expression(*src) + "))";
    } else if (opcode == "cvta.to.global.u64") {
      expression = u64_expression(*src);
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = " << expression << ";\n";
    return true;
  }

  bool emit_bfe(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    auto c = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    const char *helper =
        instruction.opcode == "bfe.s32" ? "bfe_s32" : "bfe_u32";
    body << "R[" << dst->index << "][lane] = static_cast<uint64_t>(" << helper
         << "(" << u32_expression(*a) << ", " << u32_expression(*b) << ", "
         << u32_expression(*c) << "));\n";
    return true;
  }

  bool emit_mul_hi(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_integer_operand(instruction.operand_tokens, pos);
    auto b = parse_integer_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    if (instruction.opcode == "mul.hi.u32") {
      body << "R[" << dst->index
           << "][lane] = static_cast<uint64_t>(mul_hi_u32("
           << u32_expression(*a) << ", " << u32_expression(*b) << "));\n";
    } else if (instruction.opcode == "mul.hi.s32") {
      body << "R[" << dst->index
           << "][lane] = "
              "static_cast<uint64_t>(static_cast<u32>((static_cast<int64_t>("
           << i32_expression(*a) << ") * static_cast<int64_t>("
           << i32_expression(*b) << ")) >> 32u));\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_convert_f32(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto src = parse_float_operand(instruction.operand_tokens, pos);
    if (!dst || !src) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    if (instruction.opcode == "cvt.rzi.s32.f32" ||
        instruction.opcode == "cvt.rzi.u32.f32") {
      body << "R[" << dst->index
           << "][lane] = static_cast<uint64_t>(static_cast<uint32_t>("
           << (instruction.opcode == "cvt.rzi.s32.f32"
                   ? "static_cast<int32_t>(" + f32_expression(*src) + ")"
                   : "static_cast<uint32_t>(" + f32_expression(*src) + ")")
           << "));\n";
    } else if (instruction.opcode == "cvt.rn.f32.s32" ||
               instruction.opcode == "cvt.rn.f32.u32") {
      const std::string value =
          instruction.opcode == "cvt.rn.f32.s32"
              ? "static_cast<float>(" +
                    i32_expression(Operand{Operand::Kind::Register, src->index,
                                           0, src->special}) +
                    ")"
              : "static_cast<float>(" +
                    u32_expression(Operand{Operand::Kind::Register, src->index,
                                           0, src->special}) +
                    ")";
      body << "R[" << dst->index << "][lane] = f32_to_u64(" << value << ");\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_float_binary(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_float_operand(instruction.operand_tokens, pos);
    auto b = parse_float_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string expression;
    if (opcode == "add.f32" || opcode == "add.rn.f32") {
      expression = f32_expression(*a) + " + " + f32_expression(*b);
    } else if (opcode == "sub.f32" || opcode == "sub.rn.f32") {
      expression = f32_expression(*a) + " - " + f32_expression(*b);
    } else if (opcode == "mul.f32" || opcode == "mul.rn.f32") {
      expression = f32_expression(*a) + " * " + f32_expression(*b);
    } else if (opcode == "div.full.f32" || opcode == "div.rn.f32") {
      expression = f32_expression(*a) + " / " + f32_expression(*b);
    } else if (opcode == "max.f32") {
      expression =
          "std::fmax(" + f32_expression(*a) + ", " + f32_expression(*b) + ")";
    } else if (opcode == "min.f32") {
      expression =
          "std::fmin(" + f32_expression(*a) + ", " + f32_expression(*b) + ")";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = f32_to_u64(" << expression
         << ");\n";
    return true;
  }

  bool emit_fma(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_float_operand(instruction.operand_tokens, pos);
    auto b = parse_float_operand(instruction.operand_tokens, pos);
    auto c = parse_float_operand(instruction.operand_tokens, pos);
    if (!dst || !a || !b || !c || instruction.opcode != "fma.rn.f32") {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = f32_to_u64("
         << "std::fma(" << f32_expression(*a) << ", " << f32_expression(*b)
         << ", " << f32_expression(*c) << "));\n";
    return true;
  }

  bool emit_float_unary(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_register(instruction.operand_tokens, pos);
    auto a = parse_float_operand(instruction.operand_tokens, pos);
    if (!dst || !a) {
      return false;
    }
    const std::string &opcode = instruction.opcode;
    std::string expression;
    if (opcode == "sqrt.approx.ftz.f32" || opcode == "sqrt.approx.f32") {
      expression = "std::sqrt(" + f32_expression(*a) + ")";
    } else if (opcode == "rsqrt.approx.f32" ||
               opcode == "rsqrt.approx.ftz.f32") {
      expression = "(1.0F / std::sqrt(" + f32_expression(*a) + "))";
    } else if (opcode == "abs.f32" || opcode == "abs.ftz.f32") {
      expression = "std::fabs(" + f32_expression(*a) + ")";
    } else if (opcode == "neg.f32") {
      expression = "(-" + f32_expression(*a) + ")";
    } else if (opcode == "ex2.approx.f32" || opcode == "ex2.approx.ftz.f32") {
      expression = "std::exp2(" + f32_expression(*a) + ")";
    } else {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    body << "R[" << dst->index << "][lane] = f32_to_u64(" << expression
         << ");\n";
    return true;
  }

  bool emit_predicate_binary(const FlatInstruction &instruction) {
    std::size_t pos = 0;
    auto dst = parse_predicate_index(instruction.operand_tokens, pos);
    if (!dst) {
      return false;
    }
    body_.str("");
    body_.clear();
    std::ostringstream &body = body_;
    if (instruction.opcode == "mov.pred") {
      while (pos < instruction.operand_tokens.size() &&
             instruction.operand_tokens[pos].kind == TokenKind::Comma) {
        ++pos;
      }
      if (pos < instruction.operand_tokens.size() &&
          instruction.operand_tokens[pos].kind == TokenKind::Number) {
        const bool value =
            std::stoul(instruction.operand_tokens[pos].text, nullptr, 0) != 0;
        body << "P[" << *dst << "][lane] = " << (value ? "true" : "false")
             << ";\n";
      } else {
        auto a = parse_predicate_index(instruction.operand_tokens, pos);
        if (!a) {
          return false;
        }
        body << "P[" << *dst << "][lane] = P[" << *a << "][lane];\n";
      }
    } else if (instruction.opcode == "not.pred") {
      auto a = parse_predicate_index(instruction.operand_tokens, pos);
      if (!a) {
        return false;
      }
      body << "P[" << *dst << "][lane] = !P[" << *a << "][lane];\n";
    } else if (instruction.opcode == "and.pred" ||
               instruction.opcode == "or.pred") {
      auto a = parse_predicate_index(instruction.operand_tokens, pos);
      if (!a) {
        return false;
      }
      auto b = parse_predicate_index(instruction.operand_tokens, pos);
      if (!b) {
        return false;
      }
      const char *op = instruction.opcode == "and.pred" ? "&&" : "||";
      body << "P[" << *dst << "][lane] = P[" << *a << "][lane] " << op << " P["
           << *b << "][lane];\n";
    } else {
      return false;
    }
    return true;
  }

  bool emit_instruction(std::size_t pc, const FlatInstruction &instruction) {
    const std::string &opcode = instruction.opcode;

    if (opcode == "bra" || opcode == "bra.uni") {
      return emit_branch(pc, instruction);
    }
    if (opcode == "ret" || opcode == "exit") {
      return emit_ret(pc, instruction);
    }
    if (opcode == "bar.sync") {
      return false;
    }

    body_.str("");
    body_.clear();
    bool supported = true;
    if (opcode.rfind("ld.param.", 0) == 0) {
      supported = emit_param_load(instruction);
    } else if (opcode.rfind("ld.global.", 0) == 0) {
      supported = emit_global_load(instruction);
    } else if (opcode.rfind("st.global.", 0) == 0) {
      supported = emit_global_store(instruction);
    } else if (opcode.rfind("mov.", 0) == 0) {
      if (opcode == "mov.pred") {
        supported = emit_predicate_binary(instruction);
      } else {
        supported = emit_mov(instruction);
      }
    } else if (opcode.rfind("setp.", 0) == 0) {
      const bool is_64 = opcode.find(".b64") != std::string::npos ||
                         opcode.find(".s64") != std::string::npos ||
                         opcode.find(".u64") != std::string::npos;
      if (opcode.find(".f32") != std::string::npos ||
          opcode.find(".f64") != std::string::npos) {
        supported = false;
      } else {
        supported =
            is_64 ? emit_setp_64(instruction) : emit_setp_32(instruction);
      }
    } else if (opcode == "and.pred" || opcode == "or.pred" ||
               opcode == "not.pred" || opcode == "mov.pred") {
      supported = emit_predicate_binary(instruction);
    } else if (opcode == "fma.rn.f32") {
      supported = emit_fma(instruction);
    } else if (opcode == "add.f32" || opcode == "add.rn.f32" ||
               opcode == "sub.f32" || opcode == "sub.rn.f32" ||
               opcode == "mul.f32" || opcode == "mul.rn.f32" ||
               opcode == "div.full.f32" || opcode == "div.rn.f32" ||
               opcode == "max.f32" || opcode == "min.f32") {
      supported = emit_float_binary(instruction);
    } else if (opcode == "sqrt.approx.ftz.f32" || opcode == "sqrt.approx.f32" ||
               opcode == "rsqrt.approx.f32" ||
               opcode == "rsqrt.approx.ftz.f32" || opcode == "abs.f32" ||
               opcode == "abs.ftz.f32" || opcode == "neg.f32" ||
               opcode == "ex2.approx.f32" || opcode == "ex2.approx.ftz.f32") {
      supported = emit_float_unary(instruction);
    } else if (opcode == "add.s32" || opcode == "add.u32" ||
               opcode == "sub.s32" || opcode == "sub.u32" ||
               opcode == "and.b32" || opcode == "or.b32" ||
               opcode == "xor.b32" || opcode == "mul.lo.s32" ||
               opcode == "mul.lo.u32" || opcode == "max.s32" ||
               opcode == "min.s32" || opcode == "max.u32" ||
               opcode == "min.u32") {
      supported = emit_integer_binary(instruction);
    } else if (opcode == "neg.s32" || opcode == "not.b32" ||
               opcode == "abs.s32") {
      supported = emit_integer_unary(instruction);
    } else if (opcode == "add.s64" || opcode == "add.u64" ||
               opcode == "sub.s64" || opcode == "sub.u64" ||
               opcode == "and.b64" || opcode == "or.b64" ||
               opcode == "xor.b64" || opcode == "shl.b64" ||
               opcode == "shr.u64" || opcode == "shr.s64") {
      supported = emit_integer_64(instruction);
    } else if (opcode == "shl.b32" || opcode == "shr.u32" ||
               opcode == "shr.s32") {
      supported = emit_shift_32(instruction);
    } else if (opcode == "mul.wide.s32" || opcode == "mul.wide.u32") {
      supported = emit_wide_mul(instruction);
    } else if (opcode == "mul.hi.s32" || opcode == "mul.hi.u32") {
      supported = emit_mul_hi(instruction);
    } else if (opcode == "mad.wide.s32" || opcode == "mad.wide.u32") {
      supported = emit_mad_wide(instruction);
    } else if (opcode == "mad.lo.s32") {
      supported = emit_mad_lo(instruction);
    } else if (opcode == "cvt.s64.s32" || opcode == "cvt.u64.u32" ||
               opcode == "cvt.u32.u64" || opcode == "cvta.to.global.u64") {
      supported = emit_convert(instruction);
    } else if (opcode == "cvt.rzi.s32.f32" || opcode == "cvt.rzi.u32.f32" ||
               opcode == "cvt.rn.f32.s32" || opcode == "cvt.rn.f32.u32") {
      supported = emit_convert_f32(instruction);
    } else if (opcode == "bfe.u32" || opcode == "bfe.s32") {
      supported = emit_bfe(instruction);
    } else {
      supported = false;
    }

    if (!supported) {
      return false;
    }

    // The bodies above were written into `out_` as side effects, so wrap them
    // in the predicated case structure after verifying support.
    auto predicate = predicate_expression(instruction.predicate_tokens);
    if (!predicate) {
      return false;
    }
    out_ << "          case " << pc << ": {\n";
    out_ << "            if (" << *predicate << ") {\n";
    out_ << body_.str();
    out_ << "            }\n";
    out_ << "            pc = " << (pc + 1) << ";\n";
    out_ << "            break;\n";
    out_ << "          }\n";
    return true;
  }

  const cudasim::ptx::FunctionIR &function_;
  std::unordered_map<std::string, std::size_t> parameter_offsets_;
  std::size_t next_parameter_offset_ = 0;
  std::unordered_map<std::string, std::size_t> labels_;
  std::vector<FlatInstruction> flat_;
  std::ostringstream body_;
  std::ostringstream out_;
};

} // namespace

std::optional<std::string>
generate_native_source(const cudasim::ptx::FunctionIR &function) {
  return Codegen(function).generate();
}

} // namespace cudasim::jit
