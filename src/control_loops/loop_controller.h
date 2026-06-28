#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "camera/uvc_camera_node.h"
#include "control_loops/context.h"

namespace control_loops {

class LoopController : public std::enable_shared_from_this<LoopController> {
 public:
  explicit LoopController(int num_cameras);

  void ReceiveFrame(int camera_idx, std::shared_ptr<camera::JpegBuffer> frame);

  void RegisterIterationCallback(
      int camera_idx, std::function<void(std::shared_ptr<camera::JpegBuffer>,
                                         std::shared_ptr<Context>)>
                          callback);

  void Run();
  void WakeUp();
  void RequestStop();

 private:
  int num_cameras_;

  std::vector<std::shared_ptr<camera::JpegBuffer>> latest_frames_;
  std::vector<std::mutex> frame_mutexes_;

  std::vector<std::vector<std::function<void(
      std::shared_ptr<camera::JpegBuffer>, std::shared_ptr<Context>)>>>
      callbacks_;

  std::mutex wakeup_mutex_;
  std::condition_variable wakeup_cv_;
  bool should_wake_{false};

  std::atomic<bool> stop_requested_{false};
};

}  // namespace control_loops
