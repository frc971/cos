#include "utils/cv_geometry.h"

#include <cmath>
#include <map>
#include <numbers>

#include <frc/apriltag/AprilTagFieldLayout.h>
#include <frc/geometry/Rotation3d.h>
#include <frc/geometry/Translation3d.h>
#include <opencv2/calib3d.hpp>

namespace utils {

static const cv::Mat kCvToWpilib = (cv::Mat_<double>(4, 4) << 0, 0, 1, 0, -1, 0,
                                    0, 0, 0, -1, 0, 0, 0, 0, 0, 1);
static const std::map<Basis, cv::Mat> kCvBases = {
    {Basis::kWpiToCv, kCvToWpilib.t()},
    {Basis::kCvToWpi, kCvToWpilib}};

static auto ConvertOpencvCoordinateToWpilib(cv::Mat& vec) -> void {
  const double x = vec.ptr<double>()[2];
  const double y = vec.ptr<double>()[0];
  const double z = vec.ptr<double>()[1];
  vec.ptr<double>()[0] = x;
  vec.ptr<double>()[1] = -y;
  vec.ptr<double>()[2] = -z;
}

auto CvMatToPoint3d(const cv::Mat& mat) -> cv::Point3d {
  return {mat.at<double>(0), mat.at<double>(1), mat.at<double>(2)};
}

auto HomogenizePoint3d(cv::Point3d point) -> cv::Mat {
  return (cv::Mat_<double>(4, 1) << point.x, point.y, point.z, 1.0);
}

auto QuadAreaPixels(const std::array<cv::Point2d, 4>& corners) -> double {
  return 0.5 *
         std::abs(
             (corners[0].x - corners[2].x) * (corners[1].y - corners[3].y) -
             (corners[1].x - corners[3].x) * (corners[0].y - corners[2].y));
}

auto MakeTransform(const cv::Mat& rvec, const cv::Mat& tvec) -> cv::Mat {
  CV_Assert(rvec.total() == 3 && tvec.total() == 3);
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
  rotation.copyTo(transform(cv::Rect(0, 0, 3, 3)));
  transform.at<double>(0, 3) = tvec.at<double>(0);
  transform.at<double>(1, 3) = tvec.at<double>(1);
  transform.at<double>(2, 3) = tvec.at<double>(2);
  return transform;
}

auto ChangeBasis(cv::Mat& mat, Basis basis) -> void {
  const cv::Mat& basis_mat = kCvBases.at(basis);
  mat = basis_mat * mat;
  if (mat.cols == mat.rows) {
    mat = mat * basis_mat.t();
  }
}

auto BasisMatrix(Basis basis) -> Eigen::Matrix4d {
  return CvMatToEigen(kCvBases.at(basis));
}

auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Mat& matrix)
    -> frc::Pose3d {
  cv::Mat rotation = matrix(cv::Range(0, 3), cv::Range(0, 3)).clone();
  cv::Mat tvec = matrix(cv::Range(0, 3), cv::Range(3, 4)).clone();
  cv::Mat rvec;
  cv::Rodrigues(rotation, rvec);
  ConvertOpencvCoordinateToWpilib(tvec);
  ConvertOpencvCoordinateToWpilib(rvec);
  return frc::Pose3d(CvMatToEigen(MakeTransform(rvec, tvec)));
}

auto Pose3dToCvMat(frc::Pose3d pose) -> cv::Mat {
  cv::Mat matrix = EigenToCvMat(pose.ToMatrix());
  ChangeBasis(matrix, Basis::kWpiToCv);
  return matrix;
}

auto Transform3dToCvMat(frc::Transform3d transform) -> cv::Mat {
  cv::Mat matrix = EigenToCvMat(transform.ToMatrix());
  ChangeBasis(matrix, Basis::kWpiToCv);
  return matrix;
}

}  // namespace utils
