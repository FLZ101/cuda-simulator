#include "runtime/event.h"

#include <stdexcept>

namespace cudasim::runtime {

Event::Event(std::uint64_t id) : id_(id) {}

std::uint64_t Event::id() const { return id_; }

void Event::record() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    recorded_ = true;
  }
  cv_.notify_all();
}

void Event::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return recorded_; });
}

bool Event::is_recorded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recorded_;
}

EventManager &EventManager::instance() {
  static EventManager manager;
  return manager;
}

std::shared_ptr<Event> EventManager::create() {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto handle = next_id_++;
  auto event = std::make_shared<Event>(handle);
  events_.emplace(handle, event);
  return event;
}

std::shared_ptr<Event> EventManager::get(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = events_.find(handle);
  if (it == events_.end()) {
    throw std::out_of_range("unknown event handle");
  }
  return it->second;
}

void EventManager::destroy(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.erase(handle);
}

} // namespace cudasim::runtime
