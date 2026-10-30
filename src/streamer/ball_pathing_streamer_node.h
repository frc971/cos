#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>
#include <opencv2/core.hpp>

#include "camera/camera_config.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/ball_pathing_node.h"

namespace streamer {

// Geometry and drawing shared by the live streamer and offline tests.
class BallPathingOverlay {
 public:
  BallPathingOverlay(const std::string& camera_config, double radius_m,
                     bool rotate_180 = false);
  [[nodiscard]] auto ProjectFloorPoint(cv::Point2d point) const
      -> std::optional<cv::Point2d>;
  [[nodiscard]] auto Render(const cv::Mat& image,
                            const gamepiece::BallVelocityGoalMessage* goal) const
      -> cv::Mat;

 private:
  camera::Intrinsics intrinsics_;
  camera::Extrinsics extrinsics_;
  cv::Matx44d robot_to_camera_;
  double radius_m_;
  bool rotate_180_;
};

// Consumes the JPEG and goal from the same control-loop context. Rendering is
// best effort on a dedicated worker; inference never waits for JPEG encoding.
class BallPathingStreamerNode final : public control_loop::INode {
 public:
  BallPathingStreamerNode(std::string_view jpeg_path, std::string_view goal_path,
                         const std::string& camera_config, double radius_m,
                         std::string stream_path, int port,
                         bool rotate_180 = false, double maximum_fps = 15.0);
  ~BallPathingStreamerNode() override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  void Notify(const control_loop::Context& context) const;
  nadjieb::MJPEGStreamer streamer_;
  BallPathingOverlay overlay_;
  std::string jpeg_path_;
  std::string goal_path_;
  std::string stream_path_;
  std::chrono::steady_clock::duration frame_interval_;
  std::chrono::steady_clock::time_point last_frame_{};
  std::atomic<bool> busy_{false};
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  control_loop::ThreadPool worker_{1};
};

}  // namespace streamer
