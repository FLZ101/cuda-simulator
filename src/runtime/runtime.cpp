#include "runtime/runtime.h"

namespace cudasim::runtime {

Runtime &Runtime::instance() {
  static Runtime runtime;
  return runtime;
}

DeviceRegistry &Runtime::devices() { return DeviceRegistry::instance(); }

ContextManager &Runtime::contexts() { return ContextManager::instance(); }

MemoryManager &Runtime::memory() { return MemoryManager::instance(); }

StreamManager &Runtime::streams() { return StreamManager::instance(); }

EventManager &Runtime::events() { return EventManager::instance(); }

} // namespace cudasim::runtime
