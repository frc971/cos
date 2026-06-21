#include "utils/constants_from_json.h"
#include <opencv2/core.hpp>
#include "third_party/971apriltag/apriltag.h"

namespace utils {

template <>
auto CameraMatrixFromJson<frc::apriltag::CameraMatrix>(
    nlohmann::json intrinsics) -> frc::apriltag::CameraMatrix {
  frc::apriltag::CameraMatrix camera_matrix = {.fx = intrinsics["fx"],
                                               .cx = intrinsics["cx"],
                                               .fy = intrinsics["fy"],
                                               .cy = intrinsics["cy"]};
  return camera_matrix;
}

template <>
auto DistortionCoefficientsFromJson<frc::apriltag::DistCoeffs>(
    nlohmann::json intrinsics) -> frc::apriltag::DistCoeffs {
  frc::apriltag::DistCoeffs distortion_coefficients = {.k1 = intrinsics["k1"],
                                                       .k2 = intrinsics["k2"],
                                                       .p1 = intrinsics["p1"],
                                                       .p2 = intrinsics["p2"],
                                                       .k3 = intrinsics["k3"]};
  return distortion_coefficients;
}

template <>
auto CameraMatrixFromJson<cv::Mat>(nlohmann::json intrinsics) -> cv::Mat {
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << intrinsics["fx"], 0, intrinsics["cx"], 0,
       intrinsics["fy"], intrinsics["cy"], 0, 0, 1);
  return camera_matrix;
}

template <>
auto DistortionCoefficientsFromJson<cv::Mat>(nlohmann::json intrinsics)
    -> cv::Mat {
  cv::Mat distortion_coefficients =
      (cv::Mat_<double>(1, 5) << intrinsics["k1"], intrinsics["k2"],
       intrinsics["p1"], intrinsics["p2"], intrinsics["k3"]);
  return distortion_coefficients;
}

}  // namespace utils
