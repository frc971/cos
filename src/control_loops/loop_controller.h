#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "control_loops/context.h"

namespace camera {
class JpegBuffer;
class DecodedJpegNvBuffer;
}  // namespace camera

namespace control_loops {

template <typename SyncedObjType>
class LoopController
    : public LoopControllerWakeable,
      public std::enable_shared_from_this<LoopController<SyncedObjType>> {
 public:
  explicit LoopController(int num_cameras)
      : num_cameras_(num_cameras),
        latest_frames_(num_cameras_),
        latest_metadata_(num_cameras_),
        frame_mutexes_(num_cameras_),
        callbacks_(num_cameras_) {
    for (int i = 0; i < num_cameras_; ++i) {
      latest_metadata_[i] = {MetaData{.camera_idx = i, .timestamp = 0}};
    }
  }

  void ReceiveFrame(int camera_idx, std::shared_ptr<SyncedObjType> frame,
                    unsigned long timestamp);

  void RegisterIterationCallback(
      int camera_idx,
      std::function<void(std::shared_ptr<SyncedObjType>, MetaDataList metadata,
                         std::shared_ptr<Context>)>
          callback);

  void Run();
  void WakeUp() override;
  void RequestStop();

 private:
  int num_cameras_;

  std::vector<std::shared_ptr<SyncedObjType>> latest_frames_;
  std::vector<MetaDataList> latest_metadata_;
  std::vector<std::mutex> frame_mutexes_;

  std::vector<std::vector<
      std::function<void(std::shared_ptr<SyncedObjType>, MetaDataList metadata,
                         std::shared_ptr<Context>)>>>
      callbacks_;

  std::mutex wakeup_mutex_;
  std::condition_variable wakeup_cv_;
  bool should_wake_{false};

  std::atomic<bool> stop_requested_{false};
};

template <typename SyncedObjType>
void LoopController<SyncedObjType>::ReceiveFrame(
    int camera_idx, std::shared_ptr<SyncedObjType> frame,
    unsigned long timestamp) {
  std::lock_guard<std::mutex> lock(frame_mutexes_[camera_idx]);
  latest_frames_[camera_idx] = std::move(frame);
  latest_metadata_[camera_idx] = {MetaData{
      .camera_idx = camera_idx,
      .timestamp = timestamp,
  }};
}

template <typename SyncedObjType>
void LoopController<SyncedObjType>::RegisterIterationCallback(
    int camera_idx,
    std::function<void(std::shared_ptr<SyncedObjType>, MetaDataList metadata,
                       std::shared_ptr<Context>)>
        callback) {
  callbacks_[camera_idx].push_back(std::move(callback));
}

template <typename SyncedObjType>
void LoopController<SyncedObjType>::Run() {
  while (!stop_requested_.load()) {
    std::vector<std::shared_ptr<SyncedObjType>> per_iter_frames(num_cameras_);
    std::vector<MetaDataList> per_iter_metadata(num_cameras_);

    for (int i = 0; i < num_cameras_; ++i) {
      std::lock_guard<std::mutex> lock(frame_mutexes_[i]);
      per_iter_frames[i] = latest_frames_[i];
      per_iter_metadata[i] = latest_metadata_[i];
    }

    auto ctx = std::make_shared<Context>();
    ctx->controller = this->weak_from_this();

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

template <typename SyncedObjType>
void LoopController<SyncedObjType>::WakeUp() {
  {
    std::lock_guard<std::mutex> lock(wakeup_mutex_);
    should_wake_ = true;
  }
  wakeup_cv_.notify_one();
}

template <typename SyncedObjType>
void LoopController<SyncedObjType>::RequestStop() {
  stop_requested_ = true;
  wakeup_cv_.notify_one();
}

using LocalizationLoopController = LoopController<camera::JpegBuffer>;
using GamepieceLoopController = LoopController<camera::DecodedJpegNvBuffer>;

}  // namespace control_loops
