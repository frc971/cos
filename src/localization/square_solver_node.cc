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
      camera_to_robot_(extrinsics.ToCameraToRobot<cv::Mat>()),
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

  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  cv::Mat reprojection_errors;
  cv::solvePnPGeneric(tag_corners_, detection.corners, camera_matrix_,
                      distortion_coefficients_, rvecs, tvecs, false,
                      cv::SOLVEPNP_IPPE_SQUARE, cv::noArray(), cv::noArray(),
                      reprojection_errors);

  if (rvecs.size() < 2 || tvecs.size() < 2) {
    return std::nullopt;
  }

  constexpr double kMaxUnambiguousErrorRatio = 0.2;
  const double best_error = reprojection_errors.at<double>(0);
  const double second_error = reprojection_errors.at<double>(1);
  const bool clearly_better =
      second_error > 1e-9 &&
      best_error < kMaxUnambiguousErrorRatio * second_error;

  auto build_estimate =
      [&](const cv::Mat& rvec,
          const cv::Mat& tvec) -> std::optional<solver_estimate_t> {
    const double distance = cv::norm(tvec);
    solver_estimate_t estimate;
    const cv::Mat camera_to_tag = utils::MakeTransform(rvec, tvec);
    cv::Mat tag_to_camera = camera_to_tag.inv();
    double yaw = std::atan2(tag_to_camera.at<double>(0, 2),
                            tag_to_camera.at<double>(2, 2));
    if (yaw < 0.0872665) {
      return std::nullopt;
    }
    ChangeBasis(tag_to_camera, utils::Basis::kCvToWpi);

    cv::Mat rz_flip_wpi = (cv::Mat_<double>(3, 1) << 0, 0, std::numbers::pi);
    cv::Mat empty_tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    const cv::Mat rotate_yaw_wpilib =
        utils::MakeTransform(rz_flip_wpi, empty_tvec);

    const cv::Mat field_to_tag = utils::EigenToCvMat(
        layout_.GetTagPose(detection.tag_id).value().ToMatrix());
    const cv::Mat field_to_robot =
        field_to_tag * rotate_yaw_wpilib * tag_to_camera * camera_to_robot_;
    estimate.tag_ids = {detection.tag_id};
    estimate.distances = {distance};
    estimate.pose = frc::Pose3d{utils::CvMatToEigen(field_to_robot)},
    estimate.variance = Variance(1, distance, kVarianceMin, kVarianceScalar);
    estimate.distance = distance;
    return estimate;
  };

  auto est1 = build_estimate(rvecs[0], tvecs[0]);
  auto est2 = build_estimate(rvecs[1], tvecs[1]);
  if (!est1.has_value() && !est2.has_value()) {
    return std::nullopt;
  }
  if (!est1.has_value() && est2.has_value()) {
    est1 = std::move(est2);
    est2 = std::nullopt;
  }
  if (reject_far_tags && est1->distance > kMaxTagDistance &&
      (!est2.has_value() || est2->distance > kMaxTagDistance)) {
    return std::nullopt;
  }

  return std::optional<ambiguous_estimate_t>(
      {.pos1 = std::move(est1.value()),
       .pos2 = (!est2.has_value() || clearly_better)
                   ? std::nullopt
                   : std::optional{std::move(est2)}});
}

}  // namespace localization
