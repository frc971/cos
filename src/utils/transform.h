#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <string>
#include <wpi/math/geometry/Pose3d.hpp>

namespace utils {

enum Basis { WPI_TO_CV, CV_TO_WPI };

auto MakeTransform(const cv::Mat& rvec, const cv::Mat& tvec) -> cv::Mat;

template <typename Derived>
auto EigenToCvMat(const Eigen::MatrixBase<Derived>& mat) -> cv::Mat {
  cv::Mat cvMat(mat.rows(), mat.cols(), CV_64F);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      cvMat.ptr<double>(), mat.rows(), mat.cols()) = mat;
  return cvMat;
}

auto CvMatToEigen(const cv::Mat& mat) -> Eigen::Matrix4d;

// Applies a basis change in-place. For square matrices also applies B^T on the right.
auto ChangeBasis(cv::Mat& mat, Basis basis) -> void;
auto ChangeBasis(Eigen::Matrix4d& mat, Basis basis) -> void;

auto ConvertOpencvCoordinateToWpilib(cv::Mat& vec) -> void;
auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Mat& matrix)
    -> wpi::math::Pose3d;
auto Pose3dToCvMat(wpi::math::Pose3d pose) -> cv::Mat;

auto inline Homogenize(const Eigen::Vector2d point) -> Eigen::Vector3d {
  Eigen::Vector3d homogenized_point;
  homogenized_point << point, 1;
  return homogenized_point;
}

auto inline Homogenize(const Eigen::Vector3d point) -> Eigen::Vector4d {
  Eigen::Vector4d homogenized_point;
  homogenized_point << point, 1;
  return homogenized_point;
}

struct TransformValues {
  double x;
  double y;
  double z;
  double rx;
  double ry;
  double rz;
};

struct TransformDecomposition {
  Eigen::Matrix4d translation;
  Eigen::Matrix4d Rx;
  Eigen::Matrix4d Ry;
  Eigen::Matrix4d Rz;
};

auto ExtractTranslationAndRotation(const Eigen::Matrix4d& transform_mat)
    -> TransformValues;
auto SeparateTranslationAndRotationMatrices(const TransformValues& combined)
    -> TransformDecomposition;

void PrintTransformationMatrix(
    const cv::Mat& mat,
    const std::optional<std::string>& name = std::nullopt);

inline auto PoseOffField(wpi::math::Pose3d pose) -> bool {
  constexpr double kerror_margin = 0.2;
  return pose.X().value() < 0 - kerror_margin ||
         pose.X().value() > 16.54 + kerror_margin ||
         pose.Y().value() < 0 - kerror_margin ||
         pose.Y().value() > 8 + kerror_margin;
}

}  // namespace utils
