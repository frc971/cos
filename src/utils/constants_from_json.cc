#include "utils/constants_from_json.h"
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <wpi/math/geometry/Pose3d.hpp>
#include <wpi/math/geometry/Rotation3d.hpp>
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

template <>
auto CameraMatrixFromJson<Eigen::Matrix3d>(nlohmann::json intrinsics)
    -> Eigen::Matrix3d {
  Eigen::Matrix3d K;
  K << intrinsics["fx"], 0, intrinsics["cx"], 0, intrinsics["fy"],
      intrinsics["cy"], 0, 0, 1;
  return K;
}

auto ExtrinsicsJsonToCameraToRobot(nlohmann::json extrinsics_json)
    -> wpi::math::Transform3d {
  wpi::math::Pose3d camera_pose(
      wpi::units::meter_t{extrinsics_json["translation_x"]},
      wpi::units::meter_t{extrinsics_json["translation_y"]},
      wpi::units::meter_t{extrinsics_json["translation_z"]},
      wpi::math::Rotation3d(
          wpi::units::radian_t{extrinsics_json["rotation_x"]},
          wpi::units::radian_t{extrinsics_json["rotation_y"]},
          wpi::units::radian_t{extrinsics_json["rotation_z"]}));
  wpi::math::Transform3d robot_to_camera(wpi::math::Pose3d(), camera_pose);
  return robot_to_camera.Inverse();
}

}  // namespace utils
