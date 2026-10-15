#include "localization/square_solver_node.h"

#include <utility>

#include <opencv2/calib3d.hpp>

#include "absl/log/log.h"
#include "utils/cv_geometry.h"

namespace localization {

SquareSolverNode::SquareSolverNode(std::string_view input_channel,
                                   std::string_view output_channel,
                                   const camera::Intrinsics& intrinsics,
                                   const camera::Extrinsics& extrinsics,
                                   frc::AprilTagFieldLayout layout,
                                   std::vector<cv::Point3d> tag_corners)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      layout_(std::move(layout)),
      tag_corners_(std::move(tag_corners)),
      camera_matrix_(intrinsics.ToMatrix()),
      distortion_coefficients_(intrinsics.ToDistortionCoefficients()),
      camera_to_robot_(extrinsics.ToCameraToRobot<cv::Matx44d>()),
      dependencies_({control_loop::MessageDescriptor(
          input_channel_, typeid(apriltag::TagDetections))}),
      publications_({control_loop::MessageDescriptor::Publication<
          AmbiguousEstimateMessage>(output_channel_)}) {}

void SquareSolverNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto SquareSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto notify_callbacks = [this, &context]() -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    auto* detections =
        context->GetMessage<apriltag::TagDetections>(input_channel_);
    if (detections == nullptr || detections->tag_detections.size() == 0) {
      context->SetMessage(output_channel_, nullptr);
      notify_callbacks();
      return;
    }
    if (detections->tag_detections.size() > 1) {
      LOG(WARNING) << "Misuse of square solve, multi-tag solves should be sent "
                      "to multi-tag solver. Using only the first detection";
    }

    auto estimate = AmbiguousSolve(detections->tag_detections[0]);
    if (!estimate.has_value()) {
      LOG(WARNING) << "Square solver produced no pose estimates";
      context->SetMessage(output_channel_, nullptr);
    } else {
      context->SetMessage(output_channel_,
                          std::make_unique<AmbiguousEstimateMessage>(
                              std::move(estimate.value())));
    }
    notify_callbacks();
  };
}

auto SquareSolverNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto SquareSolverNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto SquareSolverNode::AmbiguousSolve(const tag_detection_t& detection,
                                      bool reject_far_tags)
    -> std::optional<ambiguous_estimate_t> {
  if (reject_far_tags &&
      utils::QuadAreaPixels(detection.corners) < kMinTagAreaPixels) {
    return std::nullopt;
  }

  std::vector<cv::Vec3d> rvecs;
  std::vector<cv::Vec3d> tvecs;
  std::vector<double> reprojection_errors;
  cv::solvePnPGeneric(tag_corners_, detection.corners, camera_matrix_,
                      distortion_coefficients_, rvecs, tvecs, false,
                      cv::SOLVEPNP_IPPE_SQUARE, cv::noArray(), cv::noArray(),
                      reprojection_errors);

  if (rvecs.size() < 2 || tvecs.size() < 2) {
    return std::nullopt;
  }

  constexpr double kMaxUnambiguousErrorRatio = 0.2;
  const double best_error = reprojection_errors[0];
  const double second_error = reprojection_errors[1];
  const bool clearly_better =
      second_error > 1e-9 &&
      best_error < kMaxUnambiguousErrorRatio * second_error;

  auto build_estimate = [&](const cv::Vec3d& rvec,
                            const cv::Vec3d& tvec) -> solver_estimate_t {
    const double distance = cv::norm(tvec);
    solver_estimate_t estimate;
    estimate.tag_ids = {detection.tag_id};
    estimate.distances = {distance};
    estimate.pose = utils::ComputeRobotPose(tvec, rvec, detection.tag_id,
                                            layout_, camera_to_robot_);
    estimate.variance = Variance(1, distance, kVarianceMin, kVarianceScalar);
    estimate.distance = distance;
    return estimate;
  };

  auto est1 = build_estimate(rvecs[0], tvecs[0]);
  auto est2 = build_estimate(rvecs[1], tvecs[1]);
  if (reject_far_tags && est1.distance > kMaxTagDistance &&
      est2.distance > kMaxTagDistance) {
    return std::nullopt;
  }

  return std::optional<ambiguous_estimate_t>(
      {.pos1 = std::move(est1),
       .pos2 = clearly_better ? std::nullopt : std::optional{std::move(est2)}});
}

}  // namespace localization
