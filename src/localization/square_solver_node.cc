#include "localization/square_solver_node.h"
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <utility>
#include "utils/camera_utils.h"
#include "utils/constants_from_json.h"
#include "utils/transform.h"

namespace localization {

SquareSolverNode::SquareSolverNode(camera::camera_constant_t camera_constant,
                                   wpi::apriltag::AprilTagFieldLayout layout,
                                   std::vector<cv::Point3d> tag_corners)
    : SquareSolverNode(camera_constant.intrinsics_path.value(),
                       camera_constant.extrinsics_path.value(),
                       std::move(layout), std::move(tag_corners)) {}

SquareSolverNode::SquareSolverNode(const std::string& intrinsics_path,
                                   const std::string& extrinsics_path,
                                   wpi::apriltag::AprilTagFieldLayout layout,
                                   std::vector<cv::Point3d> tag_corners)
    : layout_(std::move(layout)),
      tag_corners_(std::move(tag_corners)),
      camera_matrix_(utils::CameraMatrixFromJson<cv::Mat>(
          utils::ReadIntrinsics(intrinsics_path))),
      distortion_coefficients_(utils::DistortionCoefficientsFromJson<cv::Mat>(
          utils::ReadIntrinsics(intrinsics_path))),
      camera_to_robot_(
          utils::EigenToCvMat(utils::ExtrinsicsJsonToCameraToRobot(
                                  utils::ReadExtrinsics(extrinsics_path))
                                  .ToMatrix())) {
  cv::Mat rz_flip_wpi = (cv::Mat_<double>(3, 1) << 0, 0, std::numbers::pi);
  cv::Mat empty_tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  rotate_yaw_wpilib_ = utils::MakeTransform(rz_flip_wpi, empty_tvec);
}

void SquareSolverNode::RegisterCallback(
    const std::function<void(ambiguous_estimate_t)>& callback) {
  callbacks_.push_back(callback);
}

void SquareSolverNode::AmbiguousSolve(
    std::shared_ptr<std::vector<apriltag::tag_detection_t>>& detections,
    bool reject_far_tags) {
  std::vector<ambiguous_estimate_t> pose_estimates;
  for (const auto& detection : *detections) {
    if (reject_far_tags) {
      const auto& c = detection.corners;
      const double area = 0.5 * std::abs((c[0].x - c[2].x) * (c[1].y - c[3].y) -
                                         (c[1].x - c[3].x) * (c[0].y - c[2].y));
      if (area < kmin_tag_area_pixels) {
        continue;
      }
    }

    std::vector<cv::Point2d> corners(detection.corners.begin(),
                                     detection.corners.end());
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    cv::solvePnPGeneric(tag_corners_, corners, camera_matrix_,
                        distortion_coefficients_, rvecs, tvecs, false,
                        cv::SOLVEPNP_IPPE_SQUARE);

    if (rvecs.size() < 2 || tvecs.size() < 2) {
      continue;
    }

    auto build_estimate = [&](const cv::Mat& rvec,
                              const cv::Mat& tvec) -> position_estimate_t {
      const double distance = cv::norm(tvec);
      return position_estimate_t{
          .tag_ids = {detection.tag_id},
          .rejected_tag_ids = {},
          .pose = ComputeRobotPose(tvec, rvec, detection.tag_id),
          .variance = Variance(1, distance, kvariance_min_, kvariance_scalar_),
          .timestamp = detection.timestamp,
          .num_tags = 1,
          .avg_tag_dist = distance};
    };

    auto est1 = build_estimate(rvecs[0], tvecs[0]);
    auto est2 = build_estimate(rvecs[1], tvecs[1]);

    if (reject_far_tags && est1.avg_tag_dist > kmax_tag_distance &&
        est2.avg_tag_dist > kmax_tag_distance) {
      continue;
    }

    pose_estimates.emplace_back(est1, est2);
  }
  for (const auto& pose_estimate : pose_estimates) {
    for (const auto& cb : callbacks_) {
      cb(pose_estimate);
    }
  }
}

auto SquareSolverNode::ComputeRobotPose(const cv::Mat& tvec,
                                        const cv::Mat& rvec,
                                        int tag_id) -> wpi::math::Pose3d {
  cv::Mat camera_to_tag = utils::MakeTransform(rvec, tvec);
  cv::Mat tag_to_camera = camera_to_tag.inv();
  utils::ChangeBasis(tag_to_camera, utils::CV_TO_WPI);
  cv::Mat field_to_tag = utils::EigenToCvMat(
      kapriltag_layout.GetTagPose(tag_id).value().ToMatrix());
  cv::Mat field_to_robot =
      field_to_tag * rotate_yaw_wpilib_ * tag_to_camera * camera_to_robot_;
  return wpi::math::Pose3d{utils::CvMatToEigen(field_to_robot)};
}

}  // namespace localization
