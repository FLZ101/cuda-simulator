#include "ptx/parser.h"

#include <stdexcept>
#include <utility>

namespace cudasim::ptx {

Parser::Parser(std::string source) : lexer_(std::move(source)) {
  current_ = lexer_.next();
}

Module Parser::parse() {
  Module module;

  while (current_.kind != TokenKind::End) {
    skip_newlines();
    if (current_.kind == TokenKind::End) {
      break;
    }

    if (current_.kind == TokenKind::Dot) {
      parse_directive(module);
      continue;
    }

    if (current_.kind == TokenKind::Identifier) {
      module.functions.push_back(parse_function(module));
      continue;
    }

    throw std::runtime_error("unexpected token in PTX module");
  }

  return module;
}

Token Parser::consume() {
  Token token = current_;
  current_ = lexer_.next();
  return token;
}

bool Parser::match(TokenKind kind) {
  if (current_.kind != kind) {
    return false;
  }
  consume();
  return true;
}

Token Parser::expect(TokenKind kind, const char *message) {
  if (current_.kind != kind) {
    throw std::runtime_error(message);
  }
  return consume();
}

std::string Parser::read_identifier_path() {
  std::string text = expect(TokenKind::Identifier, "expected identifier").text;
  while (current_.kind == TokenKind::Dot) {
    consume();
    text.push_back('.');
    text += expect(TokenKind::Identifier, "expected identifier after dot").text;
  }
  return text;
}

std::string Parser::read_until_semicolon() {
  std::string text;
  bool first = true;
  while (current_.kind != TokenKind::Semicolon &&
         current_.kind != TokenKind::End) {
    if (!first) {
      text.push_back(' ');
    }
    text += current_.text;
    first = false;
    consume();
  }
  expect(TokenKind::Semicolon, "expected semicolon");
  return text;
}

std::string Parser::read_until_newline() {
  std::string text;
  bool first = true;
  while (current_.kind != TokenKind::Newline &&
         current_.kind != TokenKind::End) {
    if (!first) {
      text.push_back(' ');
    }
    text += current_.text;
    first = false;
    consume();
  }
  if (current_.kind == TokenKind::Newline) {
    consume();
  }
  return text;
}

void Parser::skip_newlines() {
  while (current_.kind == TokenKind::Newline) {
    consume();
  }
}

void Parser::parse_directive(Module &module) {
  expect(TokenKind::Dot, "expected directive");
  const std::string name =
      expect(TokenKind::Identifier, "expected directive").text;

  if (name == "version") {
    module.version = read_until_newline();
    return;
  }

  if (name == "target") {
    module.target = read_until_newline();
    return;
  }

  if (name == "address_size") {
    module.address_size = read_until_newline();
    return;
  }

  if (name == "visible" || name == "extern" || name == "weak") {
    for (;;) {
      if (current_.kind == TokenKind::Dot) {
        consume();
        const std::string qualifier =
            expect(TokenKind::Identifier, "expected qualifier").text;
        if (qualifier == "entry" || qualifier == "func") {
          module.functions.push_back(parse_function(module));
          return;
        }
        continue;
      }

      if (current_.kind == TokenKind::Identifier) {
        const std::string qualifier = consume().text;
        if (qualifier == "entry" || qualifier == "func") {
          module.functions.push_back(parse_function(module));
          return;
        }
        continue;
      }

      throw std::runtime_error("expected function qualifier");
    }
  }

  if (name == "global" || name == "shared" || name == "const") {
    module.variables.push_back(parse_variable(name));
    return;
  }

  throw std::runtime_error("unsupported PTX directive: ." + name);
}

Function Parser::parse_function(Module &module) {
  Function function;

  if (current_.kind == TokenKind::Identifier) {
    function.name = consume().text;
  } else {
    function.name =
        expect(TokenKind::Identifier, "expected function name").text;
  }

  while (current_.kind != TokenKind::LBrace &&
         current_.kind != TokenKind::End) {
    if (current_.kind == TokenKind::Dot) {
      consume();
      const std::string qualifier =
          expect(TokenKind::Identifier, "expected parameter qualifier").text;
      if (qualifier != "param") {
        continue;
      }

      expect(TokenKind::Dot, "expected parameter type");
      const std::string type = read_identifier_path();
      const std::string name =
          expect(TokenKind::Identifier, "expected parameter name").text;
      std::size_t size = 8;
      if (type == "u32" || type == "b32" || type == "f32") {
        size = 4;
      } else if (type == "u16" || type == "b16" || type == "f16") {
        size = 2;
      } else if (type == "u8" || type == "b8") {
        size = 1;
      }
      function.parameters.push_back(Function::Parameter{name, type, size});
    } else {
      consume();
    }
  }
  expect(TokenKind::LBrace, "expected '{' before function body");

  while (current_.kind != TokenKind::RBrace &&
         current_.kind != TokenKind::End) {
    skip_newlines();
    if (current_.kind == TokenKind::RBrace) {
      break;
    }
    if (current_.kind == TokenKind::Dot) {
      consume();
      const std::string qualifier =
          expect(TokenKind::Identifier, "expected directive").text;
      if (qualifier == "shared" || qualifier == "local" ||
          qualifier == "const") {
        module.variables.push_back(parse_variable(qualifier));
        continue;
      }
      read_until_semicolon();
      continue;
    }
    function.instructions.push_back(parse_instruction());
  }
  expect(TokenKind::RBrace, "expected '}' after function body");

  return function;
}

Instruction Parser::parse_instruction() {
  Instruction instruction;

  if (current_.kind == TokenKind::Dollar) {
    consume();
    instruction.label = expect(TokenKind::Identifier, "expected label").text;
    expect(TokenKind::Colon, "expected ':' after label");
    skip_newlines();
  }

  if (current_.kind == TokenKind::At) {
    std::string predicate;
    consume();
    if (current_.kind == TokenKind::Exclaim) {
      predicate += "!";
      consume();
    }
    predicate += expect(TokenKind::Percent, "expected predicate register").text;
    predicate +=
        expect(TokenKind::Identifier, "expected predicate register").text;
    instruction.predicate = predicate;
  }

  if (current_.kind == TokenKind::RBrace || current_.kind == TokenKind::End) {
    return instruction;
  }

  if (current_.kind == TokenKind::Dot) {
    read_until_semicolon();
    return instruction;
  }

  instruction.opcode = read_identifier_path();
  instruction.operands += read_until_semicolon();
  if (!instruction.operands.empty() && instruction.operands.front() == ' ') {
    instruction.operands.erase(instruction.operands.begin());
  }
  return instruction;
}

VariableDecl Parser::parse_variable(const std::string &space) {
  VariableDecl variable;
  variable.space = space;

  if (current_.kind == TokenKind::Dot) {
    consume();
    const std::string first =
        expect(TokenKind::Identifier, "expected variable qualifier").text;
    if (first == "align") {
      variable.align = expect(TokenKind::Number, "expected alignment").text;
      if (current_.kind == TokenKind::Dot) {
        consume();
        variable.type =
            expect(TokenKind::Identifier, "expected variable type").text;
      }
    } else {
      variable.type = first;
    }
  } else {
    variable.type = read_identifier_path();
  }

  variable.name = expect(TokenKind::Identifier, "expected variable name").text;

  if (current_.kind == TokenKind::LBracket) {
    consume();
    variable.size = expect(TokenKind::Number, "expected array size").text;
    expect(TokenKind::RBracket, "expected ']'");
  }

  while (current_.kind != TokenKind::Semicolon &&
         current_.kind != TokenKind::End) {
    consume();
  }
  expect(TokenKind::Semicolon, "expected semicolon after variable");
  return variable;
}

} // namespace cudasim::ptx
