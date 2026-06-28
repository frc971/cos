#include "control_loops/loop_controller.h"

namespace control_loops {

Context::~Context() {
  if (auto ctrl = controller.lock()) {
    ctrl->WakeUp();
  }
}

LoopController::LoopController(int num_cameras)
    : num_cameras_(num_cameras),
      latest_frames_(num_cameras_),
      frame_mutexes_(num_cameras_),
      callbacks_(num_cameras_) {}

void LoopController::ReceiveFrame(int camera_idx,
                                  std::shared_ptr<camera::JpegBuffer> frame) {
  std::lock_guard<std::mutex> lock(frame_mutexes_[camera_idx]);
  latest_frames_[camera_idx] = std::move(frame);
}

void LoopController::RegisterIterationCallback(
    int camera_idx, std::function<void(std::shared_ptr<camera::JpegBuffer>,
                                       std::shared_ptr<Context>)>
                        callback) {
  callbacks_[camera_idx].push_back(std::move(callback));
}

void LoopController::Run() {
  while (!stop_requested_.load()) {
    std::vector<std::shared_ptr<camera::JpegBuffer>> per_iter_frames(
        num_cameras_);

    for (int i = 0; i < num_cameras_; ++i) {
      std::lock_guard<std::mutex> lock(frame_mutexes_[i]);
      per_iter_frames[i] = latest_frames_[i];
    }

    auto ctx = std::make_shared<Context>();
    ctx->controller = weak_from_this();

    bool dispatched_callback = false;
    for (int i = 0; i < num_cameras_; ++i) {
      if (per_iter_frames[i] == nullptr) {
        continue;
      }
      for (const auto& callback : callbacks_[i]) {
        dispatched_callback = true;
        callback(per_iter_frames[i], ctx);
      }
    }

    ctx.reset();
    if (!dispatched_callback) {
      WakeUp();
    }

    std::unique_lock<std::mutex> lock(wakeup_mutex_);
    wakeup_cv_.wait(lock,
                    [this] { return should_wake_ || stop_requested_.load(); });
    should_wake_ = false;
  }
}

void LoopController::WakeUp() {
  {
    std::lock_guard<std::mutex> lock(wakeup_mutex_);
    should_wake_ = true;
  }
  wakeup_cv_.notify_one();
}

void LoopController::RequestStop() {
  stop_requested_ = true;
  wakeup_cv_.notify_one();
}

}  // namespace control_loops
