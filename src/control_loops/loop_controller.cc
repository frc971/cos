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
      latest_metadata_(num_cameras_),
      frame_mutexes_(num_cameras_),
      callbacks_(num_cameras_) {
  for (int i = 0; i < num_cameras_; ++i) {
    latest_metadata_[i] = {MetaData{.camera_idx = i, .timestamp = 0}};
  }
}

void LoopController::ReceiveFrame(int camera_idx,
                                  std::shared_ptr<camera::JpegBuffer> frame,
                                  unsigned long timestamp) {
  std::lock_guard<std::mutex> lock(frame_mutexes_[camera_idx]);
  latest_frames_[camera_idx] = std::move(frame);
  latest_metadata_[camera_idx] = {MetaData{
      .camera_idx = camera_idx,
      .timestamp = timestamp,
  }};
}

void LoopController::RegisterIterationCallback(
    int camera_idx,
    std::function<void(std::shared_ptr<camera::JpegBuffer>,
                       MetaDataList metadata, std::shared_ptr<Context>)>
        callback) {
  callbacks_[camera_idx].push_back(std::move(callback));
}

void LoopController::Run() {
  while (!stop_requested_.load()) {
    std::vector<std::shared_ptr<camera::JpegBuffer>> per_iter_frames(
        num_cameras_);
    std::vector<MetaDataList> per_iter_metadata(num_cameras_);

    for (int i = 0; i < num_cameras_; ++i) {
      std::lock_guard<std::mutex> lock(frame_mutexes_[i]);
      per_iter_frames[i] = latest_frames_[i];
      per_iter_metadata[i] = latest_metadata_[i];
    }

    auto ctx = std::make_shared<Context>();
    ctx->controller = weak_from_this();

    bool dispatched_callback = false;
    for (int i = 0; i < num_cameras_; ++i) {
      for (const auto& callback : callbacks_[i]) {
        dispatched_callback = true;
        callback(per_iter_frames[i], per_iter_metadata[i], ctx);
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
