#include "localization/multi_tag_solver_node.h"
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <wpi/math/geometry/Pose3d.hpp>
#include <wpi/math/geometry/Rotation3d.hpp>
#include <wpi/math/geometry/Translation3d.hpp>
#include "absl/log/log.h"
#include "utils/camera_utils.h"
#include "utils/constants_from_json.h"
#include "utils/transform.h"

static const cv::Mat zero_vec = (cv::Mat_<double>(3, 1) << 0, 0, 0);

namespace localization {

MultiTagSolverNode::MultiTagSolverNode(
    camera::camera_constant_t camera_constant,
    const wpi::apriltag::AprilTagFieldLayout& layout,
    const std::vector<cv::Point3d>& tag_corners)
    : MultiTagSolverNode(camera_constant.intrinsics_path.value(),
                         camera_constant.extrinsics_path.value(), layout,
                         tag_corners) {}

static auto CvMatToPoint3d(cv::Mat mat) -> cv::Point3d {
  return {mat.at<double>(0), mat.at<double>(1), mat.at<double>(2)};
}

static auto HomogenizePoint3d(cv::Point3d point) -> cv::Mat {
  return (cv::Mat_<double>(4, 1) << point.x, point.y, point.z, 1);  // NOLINT
}

static auto Transform3dToCvMat(wpi::math::Transform3d transform) -> cv::Mat {
  wpi::math::Pose3d opencv_pose(
      wpi::math::Translation3d(-transform.Y(), -transform.Z(), transform.X()),
      wpi::math::Rotation3d(-transform.Rotation().Y(),
                            -transform.Rotation().Z(),
                            transform.Rotation().X()));
  return utils::EigenToCvMat(opencv_pose.ToMatrix());
}

MultiTagSolverNode::MultiTagSolverNode(
    const std::string& intrinsics_path, const std::string& extrinsics_path,
    const wpi::apriltag::AprilTagFieldLayout& layout,
    const std::vector<cv::Point3d>& tag_corners)
    : camera_matrix_(utils::CameraMatrixFromJson<cv::Mat>(
          utils::ReadIntrinsics(intrinsics_path))),
      distortion_coefficients_(utils::DistortionCoefficientsFromJson<cv::Mat>(
          utils::ReadIntrinsics(intrinsics_path))),
      camera_to_robot_(Transform3dToCvMat(utils::ExtrinsicsJsonToCameraToRobot(
          utils::ReadExtrinsics(extrinsics_path)))),
      single_tag_solver_(intrinsics_path, extrinsics_path) {
  cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0, std::numbers::pi, 0);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  cv::Mat rotate_z = utils::MakeTransform(rvec, tvec);

  for (const wpi::apriltag::AprilTag& tag : layout.GetTags()) {
    cv::Mat field_to_tag = utils::Pose3dToCvMat(tag.pose);
    tag_corners_[tag.ID] = {
        CvMatToPoint3d(field_to_tag * rotate_z *
                       HomogenizePoint3d(kapriltag_corners[0])),
        CvMatToPoint3d(field_to_tag * rotate_z *
                       HomogenizePoint3d(kapriltag_corners[1])),
        CvMatToPoint3d(field_to_tag * rotate_z *
                       HomogenizePoint3d(kapriltag_corners[2])),
        CvMatToPoint3d(field_to_tag * rotate_z *
                       HomogenizePoint3d(kapriltag_corners[3])),
    };
  }
}

void MultiTagSolverNode::RegisterCallback(
    const std::function<
        void(ambiguous_estimate_t, control_loops::MetaDataList metadata,
             std::shared_ptr<control_loops::Context>)>& callback) {
  callbacks_.emplace_back(callback);
  single_tag_solver_.RegisterCallback(callback);
}

void MultiTagSolverNode::AmbiguousSolve(
    const std::shared_ptr<std::vector<apriltag::tag_detection_t>>& detections,
    control_loops::MetaDataList metadata,
    std::shared_ptr<control_loops::Context> ctx, bool reject_far_tags) {
  if (metadata.empty()) {
    LOG(WARNING) << "MultiTagSolverNode received empty metadata";
  }
  const std::optional<ambiguous_estimate_t> pose_estimate =
      AmbiguousSolveWithoutNotify(*detections, reject_far_tags);
  if (!pose_estimate) {
    return;
  }
  for (const auto& cb : callbacks_) {
    cb(pose_estimate.value(), metadata, ctx);
  }
}

auto MultiTagSolverNode::AmbiguousSolveWithoutNotify(
    const std::vector<apriltag::tag_detection_t>& detections,
    bool reject_far_tags) -> std::optional<ambiguous_estimate_t> {
  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;
  std::vector<int> tag_ids;
  std::vector<int> rejected_tag_ids;
  double avg_distance = 0.0;

  for (const apriltag::tag_detection_t& detection : detections) {
    if (!tag_corners_[detection.tag_id].has_value()) {
      LOG(WARNING) << "Invalid tag id: " << detection.tag_id;
      continue;
    }
    if (reject_far_tags) {
      const auto& c = detection.corners;
      const double area = 0.5 * std::abs((c[0].x - c[2].x) * (c[1].y - c[3].y) -
                                         (c[1].x - c[3].x) * (c[0].y - c[2].y));
      if (area < kmin_tag_area_pixels) {
        rejected_tag_ids.push_back(detection.tag_id);
        continue;
      }
    }

    cv::Mat rvec_tag = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec_tag = cv::Mat::zeros(3, 1, CV_64FC1);
    std::vector<cv::Point2d> corners(detection.corners.begin(),
                                     detection.corners.end());
    cv::solvePnP(kapriltag_corners, corners, camera_matrix_,
                 distortion_coefficients_, rvec_tag, tvec_tag, false,
                 cv::SOLVEPNP_IPPE_SQUARE);

    if (reject_far_tags && cv::norm(tvec_tag) > kmax_tag_distance) {
      rejected_tag_ids.push_back(detection.tag_id);
      continue;
    }
    avg_distance += cv::norm(tvec_tag);
    tag_ids.push_back(detection.tag_id);
    image_points.insert(image_points.end(), detection.corners.begin(),
                        detection.corners.end());
    object_points.insert(object_points.end(),
                         tag_corners_[detection.tag_id].value().begin(),
                         tag_corners_[detection.tag_id].value().end());
  }

  if (image_points.empty() || object_points.empty()) {
    return std::nullopt;
  }

  if (tag_ids.size() == 1) {
    const std::vector<ambiguous_estimate_t> square_estimate =
        single_tag_solver_.AmbiguousSolveWithoutNotify(detections,
                                                       reject_far_tags);
    if (square_estimate.empty()) {
      return std::nullopt;
    }
    return square_estimate[0];
  }

  avg_distance /= tag_ids.size();
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

  try {
    cv::solvePnP(object_points, image_points, camera_matrix_,
                 distortion_coefficients_, rvec, tvec, false,
                 cv::SOLVEPNP_SQPNP);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught solvePnP exception:\n" << e.what();
    return std::nullopt;
  }

  cv::Mat field_to_camera = utils::MakeTransform(rvec, tvec).inv();
  cv::Mat field_to_robot = field_to_camera * camera_to_robot_;
  int num_tags = tag_ids.size();

  ambiguous_estimate_t result{
      .pos1 =
          position_estimate_t{
              .tag_ids = std::move(tag_ids),
              .rejected_tag_ids = std::move(rejected_tag_ids),
              .pose = utils::ConvertOpencvTransformationMatrixToWpilibPose(
                  field_to_robot),
              .variance = num_tags == 1
                              ? 100.0
                              : Variance(num_tags, avg_distance, kvariance_min_,
                                         kvariance_scalar_),
              .num_tags = num_tags,
              .avg_tag_dist = avg_distance},
      .pos2 = std::nullopt};
  return result;
}

}  // namespace localization
