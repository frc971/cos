#include "gamepiece/cluster_strategy.h"
#include "gamepiece/hsv_cluster_tracker_node.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace gamepiece {

YoloClusterStrategy::YoloClusterStrategy(BallDetector detector)
    : detector_(std::move(detector)) {
  if (!detector_)
    throw std::invalid_argument("A ball detector is required");
}

void YoloClusterStrategy::Process(HSVClusterTrackerNode& tracker,
                                  cv::InputArray frame,
                                  const std::optional<frc::Pose2d>& pose) {
  tracker.DetectBallClusters(frame, pose, detector_);
  for (auto& cluster : tracker.clusters_)
    tracker.UpdateDensity(cluster, cluster.ground_covar, pose);
}

void HSVClusterTrackerNode::DetectBallClusters(
    cv::InputArray frame, const std::optional<frc::Pose2d>& pose,
    const BallDetector& detector) {
  std::vector<cv::Point2f> pixels, ground;
  const auto rotation = camera_extrinsics_cv_.get_minor<3, 3>(0, 0);
  cv::cuda::GpuMat image;
  if (frame.isGpuMat())
    image = frame.getGpuMat();
  else
    image.upload(frame.getMat());
  for (const auto& pixel : detector(image)) {
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x < 0 ||
        pixel.y < 0 || pixel.x >= image.cols || pixel.y >= image.rows)
      continue;
    const cv::Point2f normalized{
        (pixel.x - camera_intrinsics_(0, 2)) / camera_intrinsics_(0, 0),
        (pixel.y - camera_intrinsics_(1, 2)) / camera_intrinsics_(1, 1)};
    const auto ray = rotation * cv::Vec3f{normalized.x, normalized.y, 1};
    if (ray[1] <= 1e-4f)
      continue;  // Camera's +Y points down after basis conversion.
    auto position = RobotRelativeOffset(normalized);
    if (!std::isfinite(position.X().value()) ||
        !std::isfinite(position.Y().value()))
      continue;
    if (pose)
      position = pose->Translation() + position.RotateBy(pose->Rotation());
    if (pose && (position.X().value() < 4.60 || position.X().value() > 11.94 ||
                 position.Y().value() < 0 || position.Y().value() > 8.07))
      continue;
    pixels.push_back(normalized);
    ground.emplace_back(position.X().value(), position.Y().value());
  }
  if (ground.empty())
    return;
  const std::vector<float> weights(ground.size(), 1.0f);
  clusters_ = ClusterGroundPoints(pixels, ground, weights, weights, pose);
  AssignPersistentIds(clusters_, pose);
  for (auto& c : clusters_) {
    c.verified_ball_centers = c.ground_points;
    c.verified_ball_count = c.ground_points.size();
    c.instantaneous_ball_count = c.verified_ball_count;
  }
}

}  // namespace gamepiece
