#include "localization/multi_tag_solver_node.h"

#include <cmath>
#include <numbers>
#include <string>
#include <utility>

#include <opencv2/calib3d.hpp>

#include "absl/log/log.h"
#include "utils/cv_geometry.h"

namespace localization {

MultiTagSolverNode::MultiTagSolverNode(
    std::string_view input_channel, std::string_view output_channel,
    const camera::Intrinsics& intrinsics, const camera::Extrinsics& extrinsics,
    const frc::AprilTagFieldLayout& layout,
    const std::vector<cv::Point3d>& tag_corners)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      image_width_(intrinsics.width),
      image_height_(intrinsics.height),
      camera_matrix_(intrinsics.ToMatrix()),
      distortion_coefficients_(intrinsics.ToDistortionCoefficients()),
      camera_to_robot_(utils::Transform3dToCvMat(
          extrinsics.ToCameraToRobot<frc::Transform3d>())),
      single_tag_solver_(input_channel, output_channel, intrinsics, extrinsics,
                         layout, tag_corners),
      dependencies_({control_loop::MessageDescriptor(
          input_channel_, typeid(apriltag::TagDetections))}),
      publications_({control_loop::MessageDescriptor::Publication<
          AmbiguousEstimateMessage>(output_channel_)}) {
  cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0, std::numbers::pi, 0);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  cv::Mat rotate_z = utils::MakeTransform(rvec, tvec);

  for (const frc::AprilTag& tag : layout.GetTags()) {
    cv::Mat field_to_tag = utils::Pose3dToCvMat(tag.pose);
    tag_corners_[tag.ID] = {
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[0])),
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[1])),
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[2])),
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[3])),
    };
  }
}

void MultiTagSolverNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto MultiTagSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto notify_callbacks = [this, &context]() -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    auto* detections =
        context->GetMessage<apriltag::TagDetections>(input_channel_);
    if (detections == nullptr) {
      context->SetMessage(output_channel_, nullptr);
      notify_callbacks();
      return;
    }

    auto estimate =
        AmbiguousSolve(detections->tag_detections, reject_far_tags_);
    if (!estimate.has_value()) {
      context->SetMessage(output_channel_, nullptr);
    } else {
      context->SetMessage(output_channel_,
                          std::make_unique<AmbiguousEstimateMessage>(
                              std::move(estimate.value())));
    }
    notify_callbacks();
  };
}

auto MultiTagSolverNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto MultiTagSolverNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void MultiTagSolverNode::SetRejectFarTags(bool reject_far_tags) {
  reject_far_tags_ = reject_far_tags;
}

auto MultiTagSolverNode::AmbiguousSolve(
    const std::vector<tag_detection_t>& detections, bool reject_far_tags)
    -> std::optional<ambiguous_estimate_t> {
  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;
  std::vector<int> tag_ids;
  std::vector<double> distances;
  std::vector<tag_detection_t> accepted_detections;
  double avg_distance = 0.0;

  for (const tag_detection_t& detection : detections) {
    if (!TagCornersInsideImage(detection, image_width_, image_height_)) {
      continue;
    }
    auto tag_corners_it = tag_corners_.find(detection.tag_id);
    if (tag_corners_it == tag_corners_.end()) {
      LOG(WARNING) << "Invalid tag id: " << detection.tag_id;
      continue;
    }
    if (reject_far_tags &&
        utils::QuadAreaPixels(detection.corners) < kMinTagAreaPixels) {
      continue;
    }

    cv::Mat rvec_tag = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec_tag = cv::Mat::zeros(3, 1, CV_64FC1);
    std::vector<cv::Point2d> corners(detection.corners.begin(),
                                     detection.corners.end());
    try {
      cv::solvePnP(kApriltagCorners, corners, camera_matrix_,
                   distortion_coefficients_, rvec_tag, tvec_tag, false,
                   cv::SOLVEPNP_IPPE_SQUARE);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Caught solvePnP exception: " << e.what();
      return std::nullopt;
    }

    if (reject_far_tags && cv::norm(tvec_tag) > kMaxTagDistance) {
      continue;
    }

    const double distance = cv::norm(tvec_tag);
    avg_distance += distance;
    tag_ids.push_back(detection.tag_id);
    distances.push_back(distance);
    accepted_detections.push_back(detection);
    image_points.insert(image_points.end(), detection.corners.begin(),
                        detection.corners.end());
    object_points.insert(object_points.end(), tag_corners_it->second.begin(),
                         tag_corners_it->second.end());
  }

  if (image_points.empty() || object_points.empty()) {
    return std::nullopt;
  }

  if (tag_ids.size() == 1) {
    std::optional<ambiguous_estimate_t> square_estimates =
        single_tag_solver_.AmbiguousSolve(accepted_detections[0],
                                          reject_far_tags);
    return square_estimates;
  }

  avg_distance /= static_cast<double>(tag_ids.size());
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

  try {
    cv::solvePnP(object_points, image_points, camera_matrix_,
                 distortion_coefficients_, rvec, tvec, false,
                 cv::SOLVEPNP_SQPNP);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught solvePnP exception: " << e.what();
    return std::nullopt;
  }

  cv::Mat field_to_camera = utils::MakeTransform(rvec, tvec).inv();
  cv::Mat field_to_robot = field_to_camera * camera_to_robot_;
  const int num_tags = static_cast<int>(tag_ids.size());

  solver_estimate_t estimate;
  estimate.tag_ids = std::move(tag_ids);
  estimate.distances = std::move(distances);
  estimate.pose =
      utils::ConvertOpencvTransformationMatrixToWpilibPose(field_to_robot);
  estimate.variance =
      Variance(num_tags, avg_distance, kVarianceMin, kVarianceScalar);
  estimate.distance = avg_distance;

  if (PoseOffField(estimate.pose)) {
    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_,
                      distortion_coefficients_, projected_points);
    double squared_error = 0.0;
    for (std::size_t i = 0; i < image_points.size(); ++i) {
      const cv::Point2d delta = image_points[i] - projected_points[i];
      squared_error += delta.dot(delta);
    }
    const double reprojection_rmse =
        std::sqrt(squared_error / static_cast<double>(image_points.size()));
    std::string tag_list;
    for (const int tag_id : estimate.tag_ids) {
      if (!tag_list.empty()) {
        tag_list += ',';
      }
      tag_list += std::to_string(tag_id);
    }
    LOG(WARNING) << "SQPnP produced a physically impossible pose from tags ["
                 << tag_list << "] with reprojection RMSE " << reprojection_rmse
                 << " px: " << estimate;
    if (reject_far_tags) {
      return std::nullopt;
    }
  }

  return ambiguous_estimate_t{.pos1 = std::move(estimate),
                              .pos2 = std::nullopt};
}

}  // namespace localization
