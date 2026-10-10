#pragma once

#include <cstdint>

namespace cudasim::runtime {

class Context {
public:
  explicit Context(int device_id);

  int device_id() const;

private:
  int device_id_;
};

class ContextManager {
public:
  static ContextManager &instance();

  Context &primary_context();

private:
  ContextManager();

  Context primary_context_;
};

} // namespace cudasim::runtime
