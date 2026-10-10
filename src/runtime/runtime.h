#pragma once

#include "runtime/context.h"
#include "runtime/device.h"
#include "runtime/event.h"
#include "runtime/memory.h"
#include "runtime/stream.h"

namespace cudasim::runtime {

class Runtime {
public:
  static Runtime &instance();

  DeviceRegistry &devices();
  ContextManager &contexts();
  MemoryManager &memory();
  StreamManager &streams();
  EventManager &events();

private:
  Runtime() = default;
};

} // namespace cudasim::runtime
