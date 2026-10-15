#include "utils/camera_config.h"

#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Rotation3d.h>
#include <units/angle.h>

namespace utils {

auto CameraMatrixFromJson(const nlohmann::json& intrinsics) -> cv::Matx33d {
  return {intrinsics.at("fx").get<double>(), 0.0,
          intrinsics.at("cx").get<double>(), 0.0,
          intrinsics.at("fy").get<double>(), intrinsics.at("cy").get<double>(),
          0.0, 0.0, 1.0};
}

auto DistortionCoefficientsFromJson(const nlohmann::json& intrinsics)
    -> cv::Vec<double, 5> {
  return {intrinsics.at("k1").get<double>(), intrinsics.at("k2").get<double>(),
          intrinsics.at("p1").get<double>(),
          intrinsics.at("p2").get<double>(), intrinsics.at("k3").get<double>()};
}

auto ExtrinsicsJsonToCameraToRobot(const nlohmann::json& extrinsics)
    -> frc::Transform3d {
  frc::Pose3d camera_pose(
      units::meter_t{extrinsics.at("translation_x").get<double>()},
      units::meter_t{extrinsics.at("translation_y").get<double>()},
      units::meter_t{extrinsics.at("translation_z").get<double>()},
      frc::Rotation3d(
          units::radian_t{units::degree_t{
              extrinsics.at("rotation_x").get<double>()}},
          units::radian_t{units::degree_t{
              extrinsics.at("rotation_y").get<double>()}},
          units::radian_t{units::degree_t{
              extrinsics.at("rotation_z").get<double>()}}));
  frc::Transform3d robot_to_camera(frc::Pose3d{}, camera_pose);
  return robot_to_camera.Inverse();
}

}  // namespace utils
