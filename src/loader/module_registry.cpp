#include "loader/module_registry.h"

#include <stdexcept>
#include <utility>

namespace cudasim::loader {

ModuleRegistry &ModuleRegistry::instance() {
  static ModuleRegistry registry;
  return registry;
}

std::uint64_t
ModuleRegistry::register_module(const cudasim::ptx::Module &module) {
  const std::uint64_t handle = next_handle_++;
  modules_.emplace(handle, module);
  return handle;
}

const cudasim::ptx::Module *
ModuleRegistry::find_module(std::uint64_t handle) const {
  const auto it = modules_.find(handle);
  return it == modules_.end() ? nullptr : &it->second;
}

const cudasim::ptx::Function *
ModuleRegistry::find_function(const std::string &name) const {
  for (const auto &[handle, module] : modules_) {
    (void)handle;
    for (const auto &function : module.functions) {
      if (function.name == name) {
        return &function;
      }
    }
  }
  return nullptr;
}

void ModuleRegistry::register_runtime_function(
    std::uint64_t module_handle, const std::string &host_name,
    const std::string &function_name) {
  runtime_functions_[host_name] =
      RegisteredFunction{module_handle, function_name, host_name};
}

const RegisteredFunction *ModuleRegistry::find_runtime_function_by_host(
    const std::string &host_name) const {
  const auto it = runtime_functions_.find(host_name);
  return it == runtime_functions_.end() ? nullptr : &it->second;
}

} // namespace cudasim::loader
