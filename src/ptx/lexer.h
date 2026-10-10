#pragma once

#include "ptx/token.h"

#include <string>

namespace cudasim::ptx {

class Lexer {
public:
  explicit Lexer(std::string source);

  Token next();
  Token peek();

private:
  void skip_whitespace_and_comments();
  Token make_token(TokenKind kind, std::size_t line, std::size_t column,
                   std::string text = {});

  std::string source_;
  std::size_t pos_ = 0;
  std::size_t line_ = 1;
  std::size_t column_ = 1;
};

} // namespace cudasim::ptx
