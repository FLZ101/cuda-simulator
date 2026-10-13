#pragma once

#include "ptx/ast.h"
#include "ptx/lexer.h"
#include "ptx/token.h"

#include <string>

namespace cudasim::ptx {

class Parser {
public:
  explicit Parser(std::string source);

  Module parse();

private:
  Token consume();
  bool match(TokenKind kind);
  Token expect(TokenKind kind, const char *message);
  std::string read_identifier_path();
  std::string read_until_semicolon();
  std::string read_until_newline();
  void skip_newlines();
  void skip_function();
  void parse_directive(Module &module);
  Function parse_function(Module &module);
  Instruction parse_instruction();
  VariableDecl parse_variable(const std::string &space);

  Lexer lexer_;
  Token current_;
};

} // namespace cudasim::ptx
