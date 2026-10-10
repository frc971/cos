#pragma once

#include <functional>
#include <string>
#include <vector>

#include <frc/apriltag/AprilTagFieldLayout.h>
#include <frc/geometry/Pose3d.h>
#include <opencv2/core/mat.hpp>

#include "camera/camera_config.h"
#include "control_loop/node.h"
#include "localization/solver_common.h"

namespace localization {

class SquareSolverNode final : public control_loop::INode {
 public:
  SquareSolverNode(std::string_view input_channel,
                   std::string_view output_channel,
                   const camera::Intrinsics& intrinsics,
                   const camera::Extrinsics& extrinsics,
                   frc::AprilTagFieldLayout layout = kApriltagLayout,
                   std::vector<cv::Point3d> tag_corners = kApriltagCorners);

  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

  auto AmbiguousSolve(const tag_detection_t& detection,
                      bool reject_far_tags = true)
      -> std::optional<ambiguous_estimate_t>;

 private:
  static constexpr double kVarianceScalar = 1.0;
  static constexpr double kVarianceMin = 0.0;

  std::string input_channel_;
  std::string output_channel_;
  frc::AprilTagFieldLayout layout_;
  std::vector<cv::Point3d> tag_corners_;
  int image_width_;
  int image_height_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat camera_to_robot_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace localization
