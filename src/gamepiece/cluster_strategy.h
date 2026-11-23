#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <frc/geometry/Pose2d.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace gamepiece {
class HSVClusterTrackerNode;
// Detector consumes a GPU image and returns centers in raw image pixels.
using BallDetector =
    std::function<std::vector<cv::Point2f>(const cv::cuda::GpuMat&)>;

class ClusterStrategy {
 public:
  virtual ~ClusterStrategy() = default;
  virtual void Process(HSVClusterTrackerNode& tracker, cv::InputArray frame,
                       const std::optional<frc::Pose2d>& pose) = 0;
};

class AnalyticalClusterStrategy final : public ClusterStrategy {
 public:
  void Process(HSVClusterTrackerNode& tracker, cv::InputArray frame,
               const std::optional<frc::Pose2d>& pose) override;
};

class YoloClusterStrategy final : public ClusterStrategy {
 public:
  explicit YoloClusterStrategy(BallDetector detector);
  void Process(HSVClusterTrackerNode& tracker, cv::InputArray frame,
               const std::optional<frc::Pose2d>& pose) override;

 private:
  BallDetector detector_;
};

class HybridClusterStrategy final : public ClusterStrategy {
 public:
  HybridClusterStrategy(BallDetector detector, float area_gain = 1);
  void Process(HSVClusterTrackerNode& tracker, cv::InputArray frame,
               const std::optional<frc::Pose2d>& pose) override;

 private:
  void AddSeededArea(HSVClusterTrackerNode& tracker, cv::InputArray frame,
                     const std::optional<frc::Pose2d>& pose);
  BallDetector detector_;
  float area_gain_;
  cv::Mat_<cv::Vec3d> geometry_;  // Fixed robot XY and sphere-area weight.
  cv::Size frame_size_;
};
}  // namespace gamepiece
