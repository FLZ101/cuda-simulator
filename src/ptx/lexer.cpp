#include "ptx/lexer.h"

#include <cctype>
#include <stdexcept>

namespace cudasim::ptx {

Lexer::Lexer(std::string source) : source_(std::move(source)) {}

Token Lexer::peek() {
  const std::size_t saved_pos = pos_;
  const std::size_t saved_line = line_;
  const std::size_t saved_column = column_;
  Token token = next();
  pos_ = saved_pos;
  line_ = saved_line;
  column_ = saved_column;
  return token;
}

Token Lexer::next() {
  skip_whitespace_and_comments();

  if (pos_ >= source_.size()) {
    return make_token(TokenKind::End, line_, column_);
  }

  const std::size_t token_line = line_;
  const std::size_t token_column = column_;
  const char ch = source_[pos_];

  if (ch == '\n') {
    ++pos_;
    ++line_;
    column_ = 1;
    return make_token(TokenKind::Newline, token_line, token_column, "\n");
  }

  if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
    std::string text;
    while (pos_ < source_.size() &&
           (std::isalnum(static_cast<unsigned char>(source_[pos_])) ||
            source_[pos_] == '_')) {
      text.push_back(source_[pos_]);
      ++pos_;
      ++column_;
    }
    return make_token(TokenKind::Identifier, token_line, token_column, text);
  }

  if (std::isdigit(static_cast<unsigned char>(ch))) {
    std::string text;
    while (pos_ < source_.size() &&
           (std::isxdigit(static_cast<unsigned char>(source_[pos_])) ||
            source_[pos_] == 'x' || source_[pos_] == 'X')) {
      text.push_back(source_[pos_]);
      ++pos_;
      ++column_;
    }
    return make_token(TokenKind::Number, token_line, token_column, text);
  }

  ++pos_;
  ++column_;

  switch (ch) {
  case '%':
    return make_token(TokenKind::Percent, token_line, token_column, "%");
  case '$':
    return make_token(TokenKind::Dollar, token_line, token_column, "$");
  case '@':
    return make_token(TokenKind::At, token_line, token_column, "@");
  case '!':
    return make_token(TokenKind::Exclaim, token_line, token_column, "!");
  case '.':
    return make_token(TokenKind::Dot, token_line, token_column, ".");
  case ',':
    return make_token(TokenKind::Comma, token_line, token_column, ",");
  case ':':
    return make_token(TokenKind::Colon, token_line, token_column, ":");
  case ';':
    return make_token(TokenKind::Semicolon, token_line, token_column, ";");
  case '{':
    return make_token(TokenKind::LBrace, token_line, token_column, "{");
  case '}':
    return make_token(TokenKind::RBrace, token_line, token_column, "}");
  case '[':
    return make_token(TokenKind::LBracket, token_line, token_column, "[");
  case ']':
    return make_token(TokenKind::RBracket, token_line, token_column, "]");
  case '(':
    return make_token(TokenKind::LParen, token_line, token_column, "(");
  case ')':
    return make_token(TokenKind::RParen, token_line, token_column, ")");
  case '+':
    return make_token(TokenKind::Plus, token_line, token_column, "+");
  case '-':
    return make_token(TokenKind::Minus, token_line, token_column, "-");
  case '*':
    return make_token(TokenKind::Star, token_line, token_column, "*");
  case '/':
    return make_token(TokenKind::Slash, token_line, token_column, "/");
  case '=':
    return make_token(TokenKind::Equals, token_line, token_column, "=");
  case '<':
    return make_token(TokenKind::Less, token_line, token_column, "<");
  case '>':
    return make_token(TokenKind::Greater, token_line, token_column, ">");
  default:
    throw std::runtime_error("unexpected character in PTX source");
  }
}

void Lexer::skip_whitespace_and_comments() {
  for (;;) {
    while (pos_ < source_.size() &&
           std::isspace(static_cast<unsigned char>(source_[pos_])) &&
           source_[pos_] != '\n') {
      ++pos_;
      ++column_;
    }

    if (pos_ + 1 < source_.size() && source_[pos_] == '/' &&
        source_[pos_ + 1] == '/') {
      while (pos_ < source_.size() && source_[pos_] != '\n') {
        ++pos_;
        ++column_;
      }
      continue;
    }

    break;
  }
}

Token Lexer::make_token(TokenKind kind, std::size_t line, std::size_t column,
                        std::string text) {
  Token token;
  token.kind = kind;
  token.line = line;
  token.column = column;
  token.text = std::move(text);
  return token;
}

} // namespace cudasim::ptx
