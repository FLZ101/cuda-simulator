#include "runtime/stream.h"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudasim::runtime {

Stream::Stream(std::uint64_t id)
    : id_(id), worker_([this] { worker_loop(); }) {}

Stream::~Stream() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

std::uint64_t Stream::id() const { return id_; }

void Stream::submit_copy(void *dst, const void *src, std::size_t bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(CopyCommand{dst, src, bytes});
    busy_ = true;
  }
  cv_.notify_one();
}

void Stream::submit_memset(void *dst, int value, std::size_t bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(MemsetCommand{dst, value, bytes});
    busy_ = true;
  }
  cv_.notify_one();
}

void Stream::submit_event(std::shared_ptr<Event> event) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(event));
    busy_ = true;
  }
  cv_.notify_one();
}

void Stream::submit_wait(std::shared_ptr<Event> event) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(WaitCommand{std::move(event)});
    busy_ = true;
  }
  cv_.notify_one();
}

void Stream::synchronize() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !busy_; });
}

bool Stream::query() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !busy_;
}

void Stream::worker_loop() {
  for (;;) {
    StreamCommand command;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty()) {
        return;
      }
      command = std::move(queue_.front());
      queue_.pop_front();
    }

    if (auto *copy = std::get_if<CopyCommand>(&command)) {
      std::memcpy(copy->dst, copy->src, copy->bytes);
    } else if (auto *memset = std::get_if<MemsetCommand>(&command)) {
      std::memset(memset->dst, memset->value, memset->bytes);
    } else if (auto *wait = std::get_if<WaitCommand>(&command)) {
      wait->event->wait();
    } else if (auto *event = std::get_if<std::shared_ptr<Event>>(&command)) {
      (*event)->record();
    }

    mark_idle_if_empty();
  }
}

void Stream::mark_idle_if_empty() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    busy_ = false;
  }
  cv_.notify_all();
}

StreamManager &StreamManager::instance() {
  static StreamManager manager;
  return manager;
}

std::shared_ptr<Stream> StreamManager::create() {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto handle = next_id_++;
  auto stream = std::make_shared<Stream>(handle);
  streams_.emplace(handle, stream);
  return stream;
}

std::shared_ptr<Stream> StreamManager::get(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = streams_.find(handle);
  if (it == streams_.end()) {
    throw std::out_of_range("unknown stream handle");
  }
  return it->second;
}

void StreamManager::destroy(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  streams_.erase(handle);
}

void StreamManager::synchronize_all() {
  std::vector<std::shared_ptr<Stream>> snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &[handle, stream] : streams_) {
      (void)handle;
      snapshot.push_back(stream);
    }
  }

  for (auto &stream : snapshot) {
    stream->synchronize();
  }
}

} // namespace cudasim::runtime
