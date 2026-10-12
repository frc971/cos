#pragma once

#include <array>
#include <type_traits>

#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Transform3d.h>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace frc {
class AprilTagFieldLayout;
}

namespace utils {

enum class Basis { kWpiToCv, kCvToWpi };

auto CvMatToPoint3d(const cv::Mat& mat) -> cv::Point3d;
auto HomogenizePoint3d(cv::Point3d point) -> cv::Mat;
auto QuadAreaPixels(const std::array<cv::Point2d, 4>& corners) -> double;
auto MakeTransform(const cv::Mat& rvec, const cv::Mat& tvec) -> cv::Mat;

template <typename Derived>
auto EigenToCvMat(const Eigen::MatrixBase<Derived>& matrix) -> cv::Mat {
  cv::Mat converted(matrix.rows(), matrix.cols(), CV_64F);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      converted.ptr<double>(), matrix.rows(), matrix.cols()) = matrix;
  return converted;
}

template <typename Matrix = Eigen::Matrix4d>
auto CvMatToEigen(const cv::Mat& matrix) -> Matrix {
  static_assert(std::is_same_v<typename Matrix::Scalar, double>);
  CV_Assert(matrix.type() == CV_64F);
  CV_Assert(Matrix::RowsAtCompileTime == Eigen::Dynamic ||
            matrix.rows == Matrix::RowsAtCompileTime);
  CV_Assert(Matrix::ColsAtCompileTime == Eigen::Dynamic ||
            matrix.cols == Matrix::ColsAtCompileTime);
  return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>(
      matrix.ptr<double>(), matrix.rows, matrix.cols);
}

auto ChangeBasis(cv::Mat& mat, Basis basis) -> void;
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

auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Mat& matrix)
    -> frc::Pose3d;
auto Pose3dToCvMat(frc::Pose3d pose) -> cv::Mat;
auto Transform3dToCvMat(frc::Transform3d transform) -> cv::Mat;

}  // namespace utils
