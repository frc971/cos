#pragma once
#include <Eigen/Core>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>
#include "apriltag/tag_detection.h"
#include "control_loops/context.h"
#include "localization/position.h"
#include "utils/node.h"

namespace localization {

using ambiguous_estimate_t = struct AmbiguousEstimate {
  position_estimate_t pos1;
  std::optional<position_estimate_t> pos2;
};

constexpr double ktag_size = 0.1651;
constexpr double kmin_tag_area_pixels = 100.0;
constexpr double kmax_tag_distance = 5.0;

inline const std::vector<cv::Point3d> kapriltag_corners = {
    {-ktag_size / 2, ktag_size / 2, 0},
    {ktag_size / 2, ktag_size / 2, 0},
    {ktag_size / 2, -ktag_size / 2, 0},
    {-ktag_size / 2, -ktag_size / 2, 0}};

inline const std::vector<Eigen::Vector3d> kapriltag_corners_eigen = {
    {-ktag_size / 2, ktag_size / 2, 0},
    {ktag_size / 2, ktag_size / 2, 0},
    {ktag_size / 2, -ktag_size / 2, 0},
    {-ktag_size / 2, -ktag_size / 2, 0}};

inline const wpi::apriltag::AprilTagFieldLayout kapriltag_layout =
    wpi::apriltag::AprilTagFieldLayout::LoadField(
        wpi::apriltag::AprilTagField::k2026RebuiltAndyMark);

inline auto Variance(int num_tags, double distance, double min_variance,
                     double scalar) -> double {
  return distance * scalar / (num_tags * num_tags) + min_variance;
}

class IPositionSolverNode : public INode<ambiguous_estimate_t> {
 public:
  void RegisterCallback(
      const std::function<void(
          ambiguous_estimate_t, control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context>)>& callback) override = 0;
  virtual void AmbiguousSolve(
      const std::shared_ptr<std::vector<apriltag::tag_detection_t>>& detections,
      control_loops::MetaDataList metadata,
      std::shared_ptr<control_loops::Context> ctx,
      bool reject_far_tags = true) = 0;
  virtual ~IPositionSolverNode() = default;
};

class IAccumulatingSolverNode : public INode<position_estimate_t> {
 public:
  void RegisterCallback(
      const std::function<void(
          position_estimate_t, control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context>)>& callback) override = 0;

  virtual void Accumulate(
      std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
      control_loops::MetaDataList metadata,
      std::shared_ptr<control_loops::Context> ctx) = 0;

  virtual ~IAccumulatingSolverNode() = default;
};

}  // namespace localization
