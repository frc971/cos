#pragma once

#include <atomic>
#include <memory>
#include <vector>

namespace control_loops {

class LoopControllerWakeable {
 public:
  virtual void WakeUp() = 0;
  virtual ~LoopControllerWakeable() = default;
};

struct MetaData {
  int camera_idx = -1;
  unsigned long timestamp = 0;
};

using MetaDataList = std::vector<MetaData>;

struct Context {
  std::atomic<bool> stop_token{false};
  std::weak_ptr<LoopControllerWakeable> controller;

  ~Context();
};

}  // namespace control_loops
