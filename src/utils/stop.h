#pragma once

#include <atomic>
#include <csignal>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include "absl/log/log.h"

namespace stop {

using namespace std::literals::chrono_literals;

constexpr std::chrono::seconds kwait_interval = 1s;
constexpr std::chrono::seconds kwait_until_kill = 10s;
std::atomic<bool> stop(false);
std::atomic<bool> registered_handler(false);
inline std::mutex on_stop_mutex;
inline std::function<void()> on_stop_callback;
inline std::atomic<bool> on_stop_called(false);

inline void SignalHandler(int signal) {
  stop = true;
}

inline void RegisterHandler(std::function<void()> on_stop = {}) {
  if (registered_handler) {
    LOG(WARNING) << "Handler has already been registred";
    return;
  }
  {
    std::lock_guard<std::mutex> lock(on_stop_mutex);
    on_stop_callback = std::move(on_stop);
  }
  std::signal(SIGINT, SignalHandler);
  // std::signal(SIGILL, SignalHandler);
  // std::signal(SIGABRT, SignalHandler);
  // std::signal(SIGFPE, SignalHandler);
  // std::signal(SIGSEGV, SignalHandler);
  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGHUP, SignalHandler);
  std::signal(SIGQUIT, SignalHandler);
  // std::signal(SIGTRAP, SignalHander);
  // std::signal(SIGKILL, SignalHandler);
  // std::signal(SIGPIPE, SignalHander);
  // std::signal(SIGALRM, SignalHander);

  std::thread([] {
    while (!stop) {
      std::this_thread::sleep_for(stop::kwait_interval);
    }
    if (!on_stop_called.exchange(true)) {
      std::function<void()> callback;
      {
        std::lock_guard<std::mutex> lock(on_stop_mutex);
        callback = on_stop_callback;
      }
      if (callback) {
        callback();
      }
    }
    std::this_thread::sleep_for(stop::kwait_until_kill);
    if (stop) {
      LOG(ERROR) << "Failed to exit cleanly";
      std::raise(SIGKILL);
    }
  }).detach();

  registered_handler = true;
}

inline void WaitUntilStop() {
  while (!stop) {
    std::this_thread::sleep_for(stop::kwait_interval);
  }
}
}  // namespace stop
