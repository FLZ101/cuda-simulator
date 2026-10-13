#include "jit/native_backend.h"

#include "jit/ptx_to_cpp.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cudasim::jit {

namespace {

using NativeEntry = int (*)(const unsigned char *, const unsigned int *,
                            const unsigned int *, unsigned int, unsigned int,
                            unsigned int);

struct NativeProgram {
  void *handle = nullptr;
  NativeEntry entry = nullptr;
};

std::size_t executor_thread_count() {
  const char *value = std::getenv("CUDA_SIM_EXEC_THREADS");
  if (value == nullptr || *value == '\0') {
    return 1;
  }
  const std::string option(value);
  if (option == "auto") {
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : static_cast<std::size_t>(count);
  }
  try {
    const long count = std::stol(option);
    if (count <= 0) {
      const unsigned int hardware = std::thread::hardware_concurrency();
      return hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
    }
    return static_cast<std::size_t>(count);
  } catch (...) {
    return 1;
  }
}

std::uint64_t fnv1a64(const std::string &text) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (char character : text) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(character));
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::filesystem::path cache_root() {
  const char *override_dir = std::getenv("CUDA_SIM_NATIVE_CACHE_DIR");
  if (override_dir != nullptr && *override_dir != '\0') {
    return std::filesystem::path(override_dir);
  }
  return std::filesystem::current_path() / ".cache" / "cudasim-native";
}

std::string quote(const std::filesystem::path &path) {
  std::string text = path.string();
  if (text.find('\'') != std::string::npos) {
    std::string escaped;
    for (char ch : text) {
      if (ch == '\'') {
        escaped += "'\\''";
      } else {
        escaped.push_back(ch);
      }
    }
    text = std::move(escaped);
  }
  return "'" + text + "'";
}

std::mutex &program_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::uint64_t, NativeProgram> &program_cache() {
  static std::unordered_map<std::uint64_t, NativeProgram> cache;
  return cache;
}

std::optional<NativeProgram> load_program(const std::string &source) {
  const std::uint64_t hash = fnv1a64(source);
  {
    std::lock_guard<std::mutex> lock(program_mutex());
    const auto it = program_cache().find(hash);
    if (it != program_cache().end() && it->second.entry != nullptr) {
      return it->second;
    }
  }

  const std::filesystem::path root = cache_root();
  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) {
    return std::nullopt;
  }

  char hash_text[32] = {};
  std::snprintf(hash_text, sizeof(hash_text), "%016llx",
                static_cast<unsigned long long>(hash));
  const std::filesystem::path cpp_path =
      root / ("kernel_" + std::string(hash_text) + ".cpp");
  const std::filesystem::path so_path =
      root / ("kernel_" + std::string(hash_text) + ".so");
  const std::filesystem::path log_path =
      root / ("kernel_" + std::string(hash_text) + ".log");
  const std::filesystem::path temporary_so =
      root / ("kernel_" + std::string(hash_text) + "." +
              std::to_string(getpid()) + "." +
              std::to_string(
                  std::hash<std::thread::id>{}(std::this_thread::get_id())) +
              ".so");

  {
    std::lock_guard<std::mutex> lock(program_mutex());
    const auto it = program_cache().find(hash);
    if (it != program_cache().end() && it->second.entry != nullptr) {
      return it->second;
    }

    if (!std::filesystem::exists(so_path)) {
      std::ofstream source_file(cpp_path);
      if (!source_file) {
        return std::nullopt;
      }
      source_file << source;
      source_file.close();

      const std::string command =
          "c++ -std=c++20 -O3 -march=native -fPIC -shared -w -o " +
          quote(temporary_so) + " " + quote(cpp_path) + " 2> " +
          quote(log_path);
      const int compile_result = std::system(command.c_str());
      if (compile_result != 0) {
        std::filesystem::remove(temporary_so);
        return std::nullopt;
      }
      std::error_code rename_error;
      std::filesystem::rename(temporary_so, so_path, rename_error);
      if (rename_error) {
        std::filesystem::remove(temporary_so);
        return std::nullopt;
      }
    }
  }

  void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::ofstream log(log_path, std::ios::app);
    if (log) {
      log << "dlopen: " << (dlerror() == nullptr ? "unknown error" : dlerror())
          << '\n';
    }
    return std::nullopt;
  }

  auto *symbol = dlsym(handle, "cudasim_native_kernel");
  if (symbol == nullptr) {
    dlclose(handle);
    return std::nullopt;
  }

  NativeProgram program;
  program.handle = handle;
  program.entry = reinterpret_cast<NativeEntry>(symbol);

  std::lock_guard<std::mutex> lock(program_mutex());
  program_cache()[hash] = program;
  return program;
}

} // namespace

bool try_execute(const cudasim::ptx::FunctionIR &function,
                 const cudasim::exec::LaunchConfig &config,
                 const std::vector<std::uint8_t> &arguments,
                 const std::vector<cudasim::ptx::VariableIR> &variables) {
  const char *disable = std::getenv("CUDA_SIM_DISABLE_NATIVE");
  if (disable != nullptr && *disable != '\0' && *disable != '0') {
    return false;
  }

  for (const auto &variable : variables) {
    if (variable.space == "shared") {
      return false;
    }
  }

  const char *exclude = std::getenv("CUDA_SIM_NATIVE_EXCLUDE");
  if (exclude != nullptr && *exclude != '\0') {
    std::string patterns(exclude);
    std::size_t begin = 0;
    while (begin < patterns.size()) {
      const std::size_t end = patterns.find(',', begin);
      const std::string pattern = end == std::string::npos
                                      ? patterns.substr(begin)
                                      : patterns.substr(begin, end - begin);
      if (!pattern.empty() &&
          function.name.find(pattern) != std::string::npos) {
        return false;
      }
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1;
    }
  }

  auto source = generate_native_source(function);
  if (!source) {
    return false;
  }

  auto program = load_program(*source);
  if (!program) {
    return false;
  }

  const std::size_t grid_x = config.grid[0];
  const std::size_t grid_y = config.grid[1];
  const std::size_t grid_z = config.grid[2];
  const std::size_t block_count = grid_x * grid_y * grid_z;
  const unsigned char *argument_data =
      arguments.empty() ? nullptr : arguments.data();

  std::atomic<bool> failed{false};
  auto run_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin;
         i < end && !failed.load(std::memory_order_relaxed); ++i) {
      const unsigned int gx = static_cast<unsigned int>(i % grid_x);
      const unsigned int gy = static_cast<unsigned int>((i / grid_x) % grid_y);
      const unsigned int gz = static_cast<unsigned int>(i / (grid_x * grid_y));
      const int result = program->entry(argument_data, config.block.data(),
                                        config.grid.data(), gx, gy, gz);
      if (result != 0) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  };

  const std::size_t thread_limit = executor_thread_count();
  const std::size_t worker_count = std::min(thread_limit, block_count);
  if (worker_count <= 1) {
    run_range(0, block_count);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t base = block_count / worker_count;
    const std::size_t remainder = block_count % worker_count;
    std::size_t begin = 0;
    for (std::size_t thread_index = 0; thread_index < worker_count;
         ++thread_index) {
      const std::size_t chunk = base + (thread_index < remainder ? 1 : 0);
      const std::size_t end = begin + chunk;
      workers.emplace_back(run_range, begin, end);
      begin = end;
    }
    for (std::thread &worker : workers) {
      worker.join();
    }
  }

  return !failed.load(std::memory_order_relaxed);
}

} // namespace cudasim::jit
