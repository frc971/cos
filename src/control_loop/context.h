#pragma once

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "control_loop/message.h"

namespace logging {
class WPILogWriter;
}

namespace control_loop {

class ControlLoop;

struct ContextInternal {
  ContextInternal(std::chrono::steady_clock::time_point start,
                  ControlLoop* control_loop, std::stop_token stop_token,
                  std::uint64_t id,
                  std::shared_ptr<logging::WPILogWriter> wpilog_writer = {});
  ~ContextInternal();

  auto Exists(const std::string& path) const -> bool {
    std::scoped_lock lock(messages_mutex_);
    return messages_.contains(path);
  }

  template <typename T>
  auto GetMessage(std::string_view path) const -> T* {
    std::scoped_lock lock(messages_mutex_);
    const auto message_it = messages_.find(std::string(path));
    if (message_it == messages_.end()) {
      return nullptr;
    }
    return dynamic_cast<T*>(message_it->second.get());
  }

  template <typename T>
  auto GetMessage(std::string& path, bool& exists) const -> T* {
    std::scoped_lock lock(messages_mutex_);
    exists = messages_.contains(path);
    const auto message_it = messages_.find(std::string(path));
    if (message_it == messages_.end()) {
      return nullptr;
    }
    return dynamic_cast<T*>(message_it->second.get());
  }

  template <typename T>
  auto GetSharedMessage(std::string_view path) const -> std::shared_ptr<T> {
    std::scoped_lock lock(messages_mutex_);
    const auto message_it = messages_.find(std::string(path));
    if (message_it == messages_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<T>(message_it->second);
  }

  void SetMessage(std::string_view path, std::shared_ptr<IMessage> message) {
    std::scoped_lock lock(messages_mutex_);
    messages_.emplace(path, std::move(message));
  }

  auto GetSize() -> size_t {
    std::scoped_lock lock(messages_mutex_);
    size_t size = 0;
    for (auto& message : messages_) {
      size += message.second->GetSize();
    }
    return size;
  }

  std::chrono::steady_clock::time_point start;
  ControlLoop* control_loop;
  std::stop_token stop_token;
  std::atomic<bool> include_in_perfomance_metrics = true;
  const std::uint64_t id;

 private:
  std::shared_ptr<logging::WPILogWriter> wpilog_writer_;
  mutable std::mutex messages_mutex_;
  std::unordered_map<std::string, std::shared_ptr<IMessage>> messages_;
};

using Context = std::shared_ptr<ContextInternal>;

}  // namespace control_loop
