#pragma once
#include <array>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <optional>
#include <string>
#include <vector>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>
#include "camera/camera_constants.h"
#include "localization/position_solver.h"
#include "localization/square_solver_node.h"

namespace localization {

static constexpr uint kmax_tags = 50;

class MultiTagSolverNode : public IPositionSolverNode {
 public:
  MultiTagSolverNode(
      const std::string& intrinsics_path, const std::string& extrinsics_path,
      const wpi::apriltag::AprilTagFieldLayout& layout = kapriltag_layout,
      const std::vector<cv::Point3d>& tag_corners = kapriltag_corners);
  MultiTagSolverNode(
      camera::camera_constant_t camera_constant,
      const wpi::apriltag::AprilTagFieldLayout& layout = kapriltag_layout,
      const std::vector<cv::Point3d>& tag_corners = kapriltag_corners);

  void RegisterCallback(
      const std::function<void(ambiguous_estimate_t)>& callback) override;
  void AmbiguousSolve(
      const std::shared_ptr<std::vector<apriltag::tag_detection_t>>& detections,
      bool reject_far_tags = true) override;
  auto AmbiguousSolveWithoutNotify(
      const std::vector<apriltag::tag_detection_t>& detections,
      bool reject_far_tags = true) -> std::optional<ambiguous_estimate_t>;

 private:
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat camera_to_robot_;
  std::array<std::optional<std::array<cv::Point3d, 4>>, kmax_tags> tag_corners_;
  SquareSolverNode single_tag_solver_;
  static constexpr double kvariance_scalar_ = 0.7;
  static constexpr double kvariance_min_ = 1.0;
  std::vector<std::function<void(ambiguous_estimate_t)>> callbacks_;
};

}  // namespace localization
