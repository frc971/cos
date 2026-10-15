#pragma once

#include <array>

#include <Eigen/Core>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Transform3d.h>

namespace frc {
class AprilTagFieldLayout;
}

namespace utils {

enum class Basis { kWpiToCv, kCvToWpi };

auto CvMatToPoint3d(const cv::Vec4d& mat) -> cv::Point3d;
auto HomogenizePoint3d(cv::Point3d point) -> cv::Vec4d;
auto QuadAreaPixels(const std::array<cv::Point2d, 4>& corners) -> double;
auto MakeTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec) -> cv::Matx44d;

template <typename Derived>
auto EigenToCvMat(const Eigen::MatrixBase<Derived>& matrix)
    -> cv::Matx<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> {
  static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic &&
                Derived::ColsAtCompileTime != Eigen::Dynamic);
  cv::Matx<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
      converted;
  for (int row = 0; row < matrix.rows(); ++row) {
    for (int col = 0; col < matrix.cols(); ++col) {
      converted(row, col) = matrix(row, col);
    }
  }
  return converted;
}

template <int rows, int cols>
auto CvMatToEigen(const cv::Matx<double, rows, cols>& matrix)
    -> Eigen::Matrix<double, rows, cols> {
  Eigen::Matrix<double, rows, cols> converted;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      converted(row, col) = matrix(row, col);
    }
  }
  return converted;
}

auto ChangeBasis(cv::Matx44d& mat, Basis basis) -> void;
auto BasisMatrix(Basis basis) -> Eigen::Matrix4d;

template <typename Derived>
auto ChangeBasis(Eigen::MatrixBase<Derived>& matrix, Basis basis) -> void {
  static_assert(Derived::RowsAtCompileTime == 4 ||
                Derived::RowsAtCompileTime == Eigen::Dynamic);
  const Eigen::Matrix4d basis_matrix = BasisMatrix(basis);
  matrix.derived() = basis_matrix * matrix.derived();
  if (matrix.cols() == matrix.rows()) {
    matrix.derived() *= basis_matrix.transpose();
  }
}

inline auto Homogenize(const Eigen::Vector2d& point) -> Eigen::Vector3d {
  return {point.x(), point.y(), 1.0};
}

inline auto Homogenize(const Eigen::Vector3d& point) -> Eigen::Vector4d {
  return {point.x(), point.y(), point.z(), 1.0};
}

inline auto CrossProduct(const Eigen::Vector3d& vector) -> Eigen::Matrix3d {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(),
      -vector.y(), vector.x(), 0.0;
  return result;
}

auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Matx44d& matrix)
    -> frc::Pose3d;
auto ComputeRobotPose(const cv::Vec3d& tvec, const cv::Vec3d& rvec, int tag_id,
                      const frc::AprilTagFieldLayout& layout,
                      const cv::Matx44d& camera_to_robot) -> frc::Pose3d;
auto Pose3dToCvMat(frc::Pose3d pose) -> cv::Matx44d;
auto Transform3dToCvMat(frc::Transform3d transform) -> cv::Matx44d;

}  // namespace utils
