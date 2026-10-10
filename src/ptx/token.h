#pragma once

#include <string>

namespace cudasim::ptx {

enum class TokenKind {
  Identifier,
  Number,
  String,
  Percent,
  Dollar,
  At,
  Exclaim,
  Dot,
  Comma,
  Colon,
  Semicolon,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  LParen,
  RParen,
  Plus,
  Minus,
  Star,
  Slash,
  Equals,
  Less,
  Greater,
  Newline,
  End,
};

struct Token {
  TokenKind kind = TokenKind::End;
  std::string text;
  std::size_t line = 0;
  std::size_t column = 0;
};

} // namespace cudasim::ptx
