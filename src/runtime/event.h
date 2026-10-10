#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace cudasim::runtime {

class Event {
public:
  explicit Event(std::uint64_t id);

  std::uint64_t id() const;
  void record();
  void wait();
  bool is_recorded() const;

private:
  std::uint64_t id_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool recorded_ = false;
};

class EventManager {
public:
  static EventManager &instance();

  std::shared_ptr<Event> create();
  std::shared_ptr<Event> get(std::uint64_t handle);
  void destroy(std::uint64_t handle);

private:
  EventManager() = default;

  std::mutex mutex_;
  std::uint64_t next_id_ = 1;
  std::unordered_map<std::uint64_t, std::shared_ptr<Event>> events_;
};

} // namespace cudasim::runtime
