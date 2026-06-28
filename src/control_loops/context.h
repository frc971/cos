#pragma once

#include <atomic>
#include <memory>
#include <vector>

namespace control_loops {

class LoopController;

struct MetaData {
  int camera_idx = -1;
  unsigned long timestamp = 0;
};

using MetaDataList = std::vector<MetaData>;

struct Context {
  std::atomic<bool> stop_token{false};
  std::weak_ptr<LoopController> controller;

  ~Context();
};

}  // namespace control_loops
