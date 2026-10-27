#include "gamepiece/cluster_strategy.h"
#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace gamepiece {

HybridClusterStrategy::HybridClusterStrategy(BallDetector detector,
                                             float area_gain)
    : detector_(std::move(detector)), area_gain_(area_gain) {
  if (!detector_ || !std::isfinite(area_gain_) || area_gain_ < 0)
    throw std::invalid_argument(
        "A detector and finite nonnegative area gain are required");
}

void HybridClusterStrategy::Process(HSVClusterTrackerNode& tracker,
                                    cv::InputArray frame,
                                    const std::optional<frc::Pose2d>& pose) {
  tracker.DetectBallClusters(frame, pose, detector_);
  AddSeededArea(tracker, frame, pose);
  for (auto& cluster : tracker.clusters_)
    tracker.UpdateDensity(cluster, cluster.ground_covar, pose);
}

void HybridClusterStrategy::AddSeededArea(
    HSVClusterTrackerNode& tracker, cv::InputArray frame,
    const std::optional<frc::Pose2d>& pose) {
  if (tracker.clusters_.empty() || area_gain_ == 0)
    return;
  cv::Mat small;
  cv::Mat1b mask;
  cv::Mat image;
  if (frame.isGpuMat()) {
    frame.getGpuMat().download(tracker.frame_);
    image = tracker.frame_;
  } else {
    image = frame.getMat();
  }
  cv::resize(image, small, {}, .5, .5, cv::INTER_AREA);
  tracker.ThresholdFuel(small, mask);
  const auto rotation = tracker.camera_extrinsics_cv_.get_minor<3, 3>(0, 0);
  constexpr float radius_m = HSVClusterTrackerNode::kFuelRadiusM;
  const float height = -tracker.camera_extrinsics_cv_(1, 3) - radius_m;
  const float fx = tracker.camera_intrinsics_(0, 0),
              fy = tracker.camera_intrinsics_(1, 1);
  const float sx = static_cast<float>(frame.size().width) / small.cols,
              sy = static_cast<float>(frame.size().height) / small.rows;
  if (frame_size_ != frame.size()) {
    frame_size_ = frame.size();
    geometry_.create(small.size());
    for (int y = 0; y < small.rows; ++y)
      for (int x = 0; x < small.cols; ++x) {
        auto& sample = geometry_(y, x);
        sample = {0, 0, 0};
        const cv::Point2f pixel{
            ((x + .5f) * sx - .5f - tracker.camera_intrinsics_(0, 2)) / fx,
            ((y + .5f) * sy - .5f - tracker.camera_intrinsics_(1, 2)) / fy};
        const cv::Vec3f ray{pixel.x, pixel.y, 1}, down = rotation * ray;
        if (down[1] <= 1e-4f)
          continue;
        const float depth = height / down[1];
        const auto robot = tracker.RobotRelativeOffset(pixel);
        const float weight = sx * sy * depth * depth /
                             (std::numbers::pi_v<float> * radius_m * radius_m *
                              fx * fy * cv::norm(ray));
        sample = {robot.X().value(), robot.Y().value(), weight};
      }
  }
  struct Moments {
    float mass{};
    cv::Vec2f sum{};
    cv::Matx22f second{};
  };
  std::vector<Moments> moments(tracker.clusters_.size());
  constexpr float radius = .65f;
  struct Seed {
    cv::Vec2f position;
    std::size_t owner;
  };
  const auto cell = [](float x) {
    return static_cast<int>(std::floor(x / radius));
  };
  const auto key = [](int x, int y) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           static_cast<uint32_t>(y);
  };
  std::unordered_map<uint64_t, std::vector<Seed>> buckets;
  for (std::size_t i = 0; i < tracker.clusters_.size(); ++i)
    for (const auto& seed : tracker.clusters_[i].ground_points)
      for (int y = cell(seed.y - radius); y <= cell(seed.y + radius); ++y)
        for (int x = cell(seed.x - radius); x <= cell(seed.x + radius); ++x)
          buckets[key(x, y)].push_back({{seed.x, seed.y}, i});
  for (int y = 0; y < mask.rows; ++y)
    for (int x = 0; x < mask.cols; ++x) {
      if (!mask.at<uchar>(y, x))
        continue;
      const auto& sample = geometry_(y, x);
      if (sample[2] == 0)
        continue;
      const cv::Vec2f position{static_cast<float>(sample[0]),
                               static_cast<float>(sample[1])};
      const auto bucket =
          buckets.find(key(cell(position[0]), cell(position[1])));
      if (bucket == buckets.end())
        continue;
      std::size_t owner = tracker.clusters_.size();
      float nearest = radius * radius;
      for (const auto& seed : bucket->second) {
        const auto d = position - seed.position;
        if (d.dot(d) < nearest) {
          nearest = d.dot(d);
          owner = seed.owner;
        }
      }
      if (owner == tracker.clusters_.size())
        continue;
      const frc::Translation2d robot{units::meter_t{sample[0]},
                                     units::meter_t{sample[1]}};
      auto p = robot;
      if (pose)
        p = pose->Translation() + p.RotateBy(pose->Rotation());
      if (pose && (p.X().value() < 4.60 || p.X().value() > 11.94 ||
                   p.Y().value() < 0 || p.Y().value() > 8.07))
        continue;
      const float weight = sample[2];
      auto& m = moments[owner];
      m.mass += weight;
      m.sum += weight * position;
      m.second += weight * cv::Matx22f{position[0] * position[0],
                                       position[0] * position[1],
                                       position[0] * position[1],
                                       position[1] * position[1]};
    }
  for (std::size_t i = 0; i < tracker.clusters_.size(); ++i) {
    auto& c = tracker.clusters_[i];
    const auto& m = moments[i];
    c.total_ball_evidence[0] = m.mass;
    c.instantaneous_ball_count =
        std::max(c.instantaneous_ball_count, area_gain_ * m.mass);
    if (m.mass < 3)
      continue;
    const auto mean = m.sum / m.mass;
    const auto covariance =
        m.second / m.mass - cv::Matx22f{mean[0] * mean[0], mean[0] * mean[1],
                                        mean[0] * mean[1], mean[1] * mean[1]};
    if (cv::determinant(covariance) >= 0)
      c.ground_covar =
          (3 * c.ground_covar + m.mass * covariance) / (3 + m.mass);
  }
}

}  // namespace gamepiece
