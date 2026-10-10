#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <frc/apriltag/AprilTagFieldLayout.h>
#include <opencv2/core/mat.hpp>

#include "control_loop/node.h"
#include "localization/square_solver_node.h"
#include "localization/solver_common.h"

namespace localization {

class MultiTagSolverNode final : public control_loop::INode {
 public:
  MultiTagSolverNode(
      std::string_view input_channel, std::string_view output_channel,
      const camera::Intrinsics& intrinsics,
      const camera::Extrinsics& extrinsics,
      const frc::AprilTagFieldLayout& layout = kApriltagLayout,
      const std::vector<cv::Point3d>& tag_corners = kApriltagCorners);

  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void SetRejectFarTags(bool reject_far_tags);

  auto AmbiguousSolve(const std::vector<tag_detection_t>& detections,
                      bool reject_far_tags = true)
      -> std::optional<ambiguous_estimate_t>;

 private:
  static constexpr double kVarianceScalar = 0.7;
  static constexpr double kVarianceMin = 1.0;

  std::string input_channel_;
  std::string output_channel_;
  int image_width_;
  int image_height_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat camera_to_robot_;
  std::unordered_map<int, std::array<cv::Point3d, 4>> tag_corners_;
  SquareSolverNode single_tag_solver_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  bool reject_far_tags_ = true;
};

}  // namespace localization
