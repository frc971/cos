#pragma once

#include <atomic>
#include <memory>

namespace control_loops {

class LoopController;

struct Context {
  std::atomic<bool> stop_token{false};
  std::weak_ptr<LoopController> controller;

  ~Context();
};

}  // namespace control_loops
