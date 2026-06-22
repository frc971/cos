#include "utils/transform.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <wpi/math/geometry/Rotation3d.hpp>
#include <wpi/math/geometry/Translation3d.hpp>

namespace utils {

namespace {
const cv::Mat kCvToWpilib =
    (cv::Mat_<double>(4, 4) << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
     1);
const std::map<Basis, cv::Mat> kCvBases = {{WPI_TO_CV, kCvToWpilib.t()},
                                           {CV_TO_WPI, kCvToWpilib}};
}  // namespace

auto MakeTransform(const cv::Mat& rvec, const cv::Mat& tvec) -> cv::Mat {
  CV_Assert(rvec.total() == 3 && tvec.total() == 3);
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
  R.copyTo(T(cv::Rect(0, 0, 3, 3)));
  T.at<double>(0, 3) = tvec.at<double>(0);
  T.at<double>(1, 3) = tvec.at<double>(1);
  T.at<double>(2, 3) = tvec.at<double>(2);
  return T;
}

auto CvMatToEigen(const cv::Mat& mat) -> Eigen::Matrix4d {
  Eigen::Matrix4d out;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      out(i, j) = mat.at<double>(i, j);
    }
  }
  return out;
}

auto ChangeBasis(cv::Mat& mat, Basis basis) -> void {
  const cv::Mat& basis_mat = kCvBases.at(basis);
  mat = basis_mat * mat;
  if (mat.cols == mat.rows) {
    mat = mat * basis_mat.t();
  }
}

auto ChangeBasis(Eigen::Matrix4d& mat, Basis basis) -> void {
  const Eigen::Matrix4d basis_mat = CvMatToEigen(kCvBases.at(basis));
  mat = basis_mat * mat;
  mat = mat * basis_mat.transpose();
}

auto ConvertOpencvCoordinateToWpilib(cv::Mat& vec) -> void {
  const double x = vec.ptr<double>()[2];
  const double y = vec.ptr<double>()[0];
  const double z = vec.ptr<double>()[1];
  vec.ptr<double>()[0] = x;
  vec.ptr<double>()[1] = -y;
  vec.ptr<double>()[2] = -z;
}

auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Mat& transform)
    -> wpi::math::Pose3d {
  cv::Mat R = transform(cv::Range(0, 3), cv::Range(0, 3)).clone();
  cv::Mat tvec = transform(cv::Range(0, 3), cv::Range(3, 4)).clone();
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  ConvertOpencvCoordinateToWpilib(tvec);
  ConvertOpencvCoordinateToWpilib(rvec);
  cv::Mat wpilib_transform = MakeTransform(rvec, tvec);
  return wpi::math::Pose3d(CvMatToEigen(wpilib_transform));
}

auto Pose3dToCvMat(wpi::math::Pose3d pose) -> cv::Mat {
  wpi::math::Pose3d opencv_pose(
      wpi::math::Translation3d(-pose.Y(), -pose.Z(), pose.X()),
      wpi::math::Rotation3d(-pose.Rotation().Y(), -pose.Rotation().Z(),
                            pose.Rotation().X()));
  return EigenToCvMat(opencv_pose.ToMatrix());
}

template cv::Mat EigenToCvMat<Eigen::Matrix<double, 4, 4>>(
    const Eigen::MatrixBase<Eigen::Matrix<double, 4, 4>>&);

auto ExtractTranslationAndRotation(const Eigen::Matrix4d& transform_mat)
    -> TransformValues {
  double x = transform_mat(0, 3);
  double y = transform_mat(1, 3);
  double z = transform_mat(2, 3);
  const Eigen::Matrix3d R = transform_mat.block<3, 3>(0, 0);
  double sy = std::hypot(R(0, 0), R(1, 0));
  bool singular = sy < 1e-6;
  double roll, pitch, yaw;
  if (!singular) {
    roll = std::atan2(R(2, 1), R(2, 2));
    pitch = std::atan2(-R(2, 0), sy);
    yaw = std::atan2(R(1, 0), R(0, 0));
  } else {
    roll = std::atan2(-R(1, 2), R(1, 1));
    pitch = std::atan2(-R(2, 0), sy);
    yaw = 0.0;
  }
  return {x, y, z, roll, pitch, yaw};
}

auto SeparateTranslationAndRotationMatrices(
    const TransformValues& decomposition) -> TransformDecomposition {
  // clang-format off
  const Eigen::Matrix4d Rx = (Eigen::Matrix4d() <<
      1, 0, 0, 0,
      0,  cos(decomposition.rx), -sin(decomposition.rx), 0,
      0,  sin(decomposition.rx),  cos(decomposition.rx), 0,
      0, 0, 0, 1).finished();
  const Eigen::Matrix4d Ry = (Eigen::Matrix4d() <<
       cos(decomposition.ry), 0, sin(decomposition.ry), 0,
       0, 1, 0, 0,
      -sin(decomposition.ry), 0, cos(decomposition.ry), 0,
       0, 0, 0, 1).finished();
  const Eigen::Matrix4d Rz = (Eigen::Matrix4d() <<
      cos(decomposition.rz), -sin(decomposition.rz), 0, 0,
      sin(decomposition.rz),  cos(decomposition.rz), 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1).finished();
  // clang-format on
  Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
  translation(0, 3) = decomposition.x;
  translation(1, 3) = decomposition.y;
  translation(2, 3) = decomposition.z;
  return {translation, Rx, Ry, Rz};
}

void PrintTransformationMatrix(const cv::Mat& T,
                               const std::optional<std::string>& name) {
  CV_Assert(T.rows == 4 && T.cols == 4);
  CV_Assert(T.type() == CV_64F || T.type() == CV_32F);
  double x = T.at<double>(0, 3);
  double y = T.at<double>(1, 3);
  double z = T.at<double>(2, 3);
  cv::Mat R = T(cv::Rect(0, 0, 3, 3));
  double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                        R.at<double>(1, 0) * R.at<double>(1, 0));
  bool singular = sy < 1e-6;
  double roll, pitch, yaw;
  if (!singular) {
    roll = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    pitch = std::atan2(-R.at<double>(2, 0), sy);
    yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
  } else {
    roll = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    pitch = std::atan2(-R.at<double>(2, 0), sy);
    yaw = 0.0;
  }
  roll *= 180.0 / CV_PI;
  pitch *= 180.0 / CV_PI;
  yaw *= 180.0 / CV_PI;
  std::cout << name.value_or("Transformation Matrix") << "-> "
            << "X: " << x << " m, Y: " << y << " m, Z: " << z << " m, "
            << "Roll: " << roll << "°, Pitch: " << pitch << "°, Yaw: " << yaw
            << "°" << std::endl;
}

}  // namespace utils
