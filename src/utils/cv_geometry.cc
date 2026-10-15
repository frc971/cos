#include "utils/cv_geometry.h"

#include <cmath>
#include <map>
#include <numbers>

#include <opencv2/calib3d.hpp>
#include <frc/apriltag/AprilTagFieldLayout.h>
#include <frc/geometry/Rotation3d.h>
#include <frc/geometry/Translation3d.h>

namespace utils {

static const cv::Matx44d kCvToWpilib{
    0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1};
static const std::map<Basis, cv::Matx44d> kCvBases = {
    {Basis::kWpiToCv, kCvToWpilib.t()}, {Basis::kCvToWpi, kCvToWpilib}};

static auto ConvertOpencvCoordinateToWpilib(cv::Vec3d& vec) -> void {
  vec = {vec[2], -vec[0], -vec[1]};
}

auto CvMatToPoint3d(const cv::Vec4d& mat) -> cv::Point3d {
  return {mat[0], mat[1], mat[2]};
}

auto HomogenizePoint3d(cv::Point3d point) -> cv::Vec4d {
  return {point.x, point.y, point.z, 1.0};
}

auto QuadAreaPixels(const std::array<cv::Point2d, 4>& corners) -> double {
  return 0.5 * std::abs((corners[0].x - corners[2].x) *
                            (corners[1].y - corners[3].y) -
                        (corners[1].x - corners[3].x) *
                            (corners[0].y - corners[2].y));
}

auto MakeTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec) -> cv::Matx44d {
  cv::Matx33d rotation;
  cv::Rodrigues(rvec, rotation);
  return {rotation(0, 0), rotation(0, 1), rotation(0, 2), tvec[0],
          rotation(1, 0), rotation(1, 1), rotation(1, 2), tvec[1],
          rotation(2, 0), rotation(2, 1), rotation(2, 2), tvec[2],
          0.0, 0.0, 0.0, 1.0};
}

auto ChangeBasis(cv::Matx44d& mat, Basis basis) -> void {
  const cv::Matx44d& basis_mat = kCvBases.at(basis);
  mat = basis_mat * mat * basis_mat.t();
}

auto BasisMatrix(Basis basis) -> Eigen::Matrix4d {
  return CvMatToEigen(kCvBases.at(basis));
}

auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Matx44d& matrix)
    -> frc::Pose3d {
  const cv::Matx33d rotation = matrix.get_minor<3, 3>(0, 0);
  cv::Vec3d tvec{matrix(0, 3), matrix(1, 3), matrix(2, 3)};
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  ConvertOpencvCoordinateToWpilib(tvec);
  ConvertOpencvCoordinateToWpilib(rvec);
  return frc::Pose3d(CvMatToEigen(MakeTransform(rvec, tvec)));
}

auto ComputeRobotPose(const cv::Vec3d& tvec, const cv::Vec3d& rvec, int tag_id,
                      const frc::AprilTagFieldLayout& layout,
                      const cv::Matx44d& camera_to_robot) -> frc::Pose3d {
  const cv::Matx44d camera_to_tag = MakeTransform(rvec, tvec);
  cv::Matx44d tag_to_camera = camera_to_tag.inv();
  ChangeBasis(tag_to_camera, Basis::kCvToWpi);

  const cv::Vec3d rz_flip_wpi{0, 0, std::numbers::pi};
  const cv::Vec3d empty_tvec{0, 0, 0};
  const cv::Matx44d rotate_yaw_wpilib = MakeTransform(rz_flip_wpi, empty_tvec);

  const cv::Matx44d field_to_tag =
      EigenToCvMat(layout.GetTagPose(tag_id).value().ToMatrix());
  const cv::Matx44d field_to_robot =
      field_to_tag * rotate_yaw_wpilib * tag_to_camera * camera_to_robot;
  return frc::Pose3d{CvMatToEigen(field_to_robot)};
}

auto Pose3dToCvMat(frc::Pose3d pose) -> cv::Matx44d {
  cv::Matx44d matrix = EigenToCvMat(pose.ToMatrix());
  ChangeBasis(matrix, Basis::kWpiToCv);
  return matrix;
}

auto Transform3dToCvMat(frc::Transform3d transform) -> cv::Matx44d {
  cv::Matx44d matrix = EigenToCvMat(transform.ToMatrix());
  ChangeBasis(matrix, Basis::kWpiToCv);
  return matrix;
}

}  // namespace utils
