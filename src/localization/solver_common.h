#pragma once

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <frc/apriltag/AprilTagFieldLayout.h>
#include <frc/geometry/Pose3d.h>
#include <opencv2/core/types.hpp>

#include "apriltag/nvidia_apriltag_detector_node.h"
#include "localization/position.h"
#include "logging/log_registration.h"

namespace localization {

using tag_detection_t = apriltag::TagDetections::tag_detection;

struct SolverEstimate {
  std::vector<int> tag_ids;
  std::vector<double> distances;
  frc::Pose3d pose;
  double variance = 0.0;
  double distance = 0.0;
  LOG_FIELDS(SolverEstimate, tag_ids, distances, pose, variance, distance)

  friend auto operator<<(std::ostream& os, const SolverEstimate& estimate)
      -> std::ostream& {
    const auto& translation = estimate.pose.Translation();
    const auto& rotation = estimate.pose.Rotation();
    return os << "pose(x=" << translation.X().value()
              << " y=" << translation.Y().value()
              << " z=" << translation.Z().value()
              << " roll=" << rotation.X().value()
              << " pitch=" << rotation.Y().value()
              << " yaw=" << rotation.Z().value() << ")"
              << " variance=" << estimate.variance
              << " distance=" << estimate.distance;
  }
};

using solver_estimate_t = SolverEstimate;

struct AmbiguousEstimate {
  solver_estimate_t pos1;
  std::optional<solver_estimate_t> pos2;
  LOG_FIELDS(AmbiguousEstimate, pos1, pos2)
};

using ambiguous_estimate_t = AmbiguousEstimate;

class AmbiguousEstimateMessage final : public control_loop::IMessage {
 public:
  explicit AmbiguousEstimateMessage(AmbiguousEstimate estimate)
      : estimate(std::move(estimate)) {}

  auto GetType() -> const std::type_info& override {
    return typeid(AmbiguousEstimateMessage);
  }
  auto GetSize() -> std::size_t override {
    return sizeof(*this);
  }

  AmbiguousEstimate estimate;
  LOG_FIELDS(AmbiguousEstimateMessage, estimate)
};

inline auto DetectionBatchChannel(std::string_view camera_name) -> std::string {
  return "localization/" + std::string(camera_name) + "_detection_batch";
}

constexpr double kTagSize = 0.1651;
constexpr double kMinTagAreaPixels = 100.0;
constexpr double kMaxTagDistance = 5.0;

inline const std::vector<cv::Point3d> kApriltagCorners = {
    {-kTagSize / 2.0, kTagSize / 2.0, 0.0},
    {kTagSize / 2.0, kTagSize / 2.0, 0.0},
    {kTagSize / 2.0, -kTagSize / 2.0, 0.0},
    {-kTagSize / 2.0, -kTagSize / 2.0, 0.0}};

inline const frc::AprilTagFieldLayout kApriltagLayout =
    frc::AprilTagFieldLayout::LoadField(
        frc::AprilTagField::k2026RebuiltAndyMark);

auto Variance(int num_tags, double distance, double min_variance, double scalar)
    -> double;
auto PoseOffField(frc::Pose3d pose) -> bool;
// Leave room for the tag's outer border: a clipped quad can decode correctly
// while its fitted corners produce a wildly incorrect pose.
auto TagCornersInsideImage(const tag_detection_t& detection, int width,
                           int height) -> bool;

}  // namespace localization
