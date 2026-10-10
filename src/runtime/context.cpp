#include "runtime/context.h"

namespace cudasim::runtime {

Context::Context(int device_id) : device_id_(device_id) {}

int Context::device_id() const { return device_id_; }

ContextManager::ContextManager() : primary_context_(0) {}

ContextManager &ContextManager::instance() {
  static ContextManager manager;
  return manager;
}

Context &ContextManager::primary_context() { return primary_context_; }

} // namespace cudasim::runtime
