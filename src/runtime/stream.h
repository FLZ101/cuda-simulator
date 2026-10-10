#pragma once

#include "runtime/event.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <variant>

namespace cudasim::runtime {

struct CopyCommand {
  void *dst;
  const void *src;
  std::size_t bytes;
};

struct MemsetCommand {
  void *dst;
  int value;
  std::size_t bytes;
};

struct WaitCommand {
  std::shared_ptr<Event> event;
};

using StreamCommand = std::variant<CopyCommand, MemsetCommand, WaitCommand,
                                   std::shared_ptr<Event>>;

class Stream {
public:
  explicit Stream(std::uint64_t id);
  ~Stream();

  std::uint64_t id() const;
  void submit_copy(void *dst, const void *src, std::size_t bytes);
  void submit_memset(void *dst, int value, std::size_t bytes);
  void submit_event(std::shared_ptr<Event> event);
  void submit_wait(std::shared_ptr<Event> event);
  void synchronize();
  bool query() const;

private:
  void worker_loop();
  void mark_idle_if_empty();

  std::uint64_t id_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<StreamCommand> queue_;
  bool busy_ = false;
  bool stop_ = false;
  std::thread worker_;
};

class StreamManager {
public:
  static StreamManager &instance();

  std::shared_ptr<Stream> create();
  std::shared_ptr<Stream> get(std::uint64_t handle);
  void destroy(std::uint64_t handle);
  void synchronize_all();

private:
  StreamManager() = default;

  std::mutex mutex_;
  std::uint64_t next_id_ = 1;
  std::unordered_map<std::uint64_t, std::shared_ptr<Stream>> streams_;
};

} // namespace cudasim::runtime
