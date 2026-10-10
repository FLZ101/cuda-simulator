#pragma once

#include "ptx/ast.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudasim::loader {

struct RegisteredFunction {
  std::uint64_t module_handle = 0;
  std::string function_name;
  std::string host_name;
};

class ModuleRegistry {
public:
  static ModuleRegistry &instance();

  std::uint64_t register_module(const cudasim::ptx::Module &module);
  const cudasim::ptx::Module *find_module(std::uint64_t handle) const;
  const cudasim::ptx::Function *find_function(const std::string &name) const;
  void register_runtime_function(std::uint64_t module_handle,
                                 const std::string &host_name,
                                 const std::string &function_name);
  const RegisteredFunction *
  find_runtime_function_by_host(const std::string &host_name) const;

private:
  ModuleRegistry() = default;

  std::uint64_t next_handle_ = 1;
  std::unordered_map<std::uint64_t, cudasim::ptx::Module> modules_;
  std::unordered_map<std::string, RegisteredFunction> runtime_functions_;
};

} // namespace cudasim::loader
