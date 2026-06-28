#pragma once
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>
#include <wpi/math/geometry/Pose3d.hpp>
#include "camera/camera_constants.h"
#include "localization/position_solver.h"

namespace localization {

class SquareSolverNode : public IPositionSolverNode {
 public:
  SquareSolverNode(const std::string& intrinsics_path,
                   const std::string& extrinsics_path,
                   wpi::apriltag::AprilTagFieldLayout layout = kapriltag_layout,
                   std::vector<cv::Point3d> tag_corners = kapriltag_corners);
  SquareSolverNode(camera::camera_constant_t camera_constant,
                   wpi::apriltag::AprilTagFieldLayout layout = kapriltag_layout,
                   std::vector<cv::Point3d> tag_corners = kapriltag_corners);

  void RegisterCallback(
      const std::function<
          void(ambiguous_estimate_t, control_loops::MetaDataList metadata,
               std::shared_ptr<control_loops::Context>)>& callback) override;
  void AmbiguousSolve(
      const std::shared_ptr<std::vector<apriltag::tag_detection_t>>& detections,
      control_loops::MetaDataList metadata,
      std::shared_ptr<control_loops::Context> ctx,
      bool reject_far_tags = true) override;
  auto AmbiguousSolveWithoutNotify(
      const std::vector<apriltag::tag_detection_t>& detections,
      bool reject_far_tags = true) -> std::vector<ambiguous_estimate_t>;

 private:
  auto ComputeRobotPose(const cv::Mat& tvec, const cv::Mat& rvec,
                        int tag_id) -> wpi::math::Pose3d;

  static constexpr double kvariance_scalar_ = 1.0;
  static constexpr double kvariance_min_ = 0.0;
  wpi::apriltag::AprilTagFieldLayout layout_;
  std::vector<cv::Point3d> tag_corners_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat camera_to_robot_;
  cv::Mat rotate_yaw_wpilib_;
  std::vector<std::function<void(ambiguous_estimate_t,
                                 control_loops::MetaDataList metadata,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;
};

}  // namespace localization
