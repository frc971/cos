#include "camera/camera_config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <units/angle.h>

#include "absl/log/check.h"

#include "utils/cv_geometry.h"

namespace camera {
Intrinsics::Intrinsics(const std::filesystem::path& path) {
  CHECK(std::filesystem::is_regular_file(path));
  std::ifstream intrinsics_file(path);
  CHECK(intrinsics_file.is_open());
  nlohmann::json json;
  intrinsics_file >> json;

  width = json.at("width").get<int>();
  height = json.at("height").get<int>();
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  auto intrinsics_json = json.at("intrinsics");

  cx = intrinsics_json.at("cx").get<double>();
  cy = intrinsics_json.at("cy").get<double>();
  fx = intrinsics_json.at("fx").get<double>();
  fy = intrinsics_json.at("fy").get<double>();

  k1 = intrinsics_json.value("k1", 0.0);
  k2 = intrinsics_json.value("k2", 0.0);
  k3 = intrinsics_json.value("k3", 0.0);
  p1 = intrinsics_json.value("p1", 0.0);
  p2 = intrinsics_json.value("p2", 0.0);
}

auto Intrinsics::ToMatrix() const -> cv::Mat {
  cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);

  matrix.at<double>(0, 0) = fx;
  matrix.at<double>(0, 2) = cx;
  matrix.at<double>(1, 1) = fy;
  matrix.at<double>(1, 2) = cy;

  return matrix;
}

[[nodiscard]] auto Intrinsics::ToDistortionCoefficients() const -> cv::Mat {
  return cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3;  // NOLINT
}

Extrinsics::Extrinsics(const std::filesystem::path& path) {
  CHECK(std::filesystem::is_regular_file(path));
  std::ifstream extrinsics_file(path);
  CHECK(extrinsics_file.is_open());
  nlohmann::json json;
  extrinsics_file >> json;

  auto extrinsics_json = json.at("extrinsics");

  translation_x = extrinsics_json.at("translation_x").get<double>();
  translation_y = extrinsics_json.at("translation_y").get<double>();
  translation_z = extrinsics_json.at("translation_z").get<double>();

  rotation_x = units::radian_t{units::degree_t{
      extrinsics_json.at("rotation_x").get<double>()}}.value();
  rotation_y = units::radian_t{units::degree_t{
      extrinsics_json.at("rotation_y").get<double>()}}.value();
  rotation_z = units::radian_t{units::degree_t{
      extrinsics_json.at("rotation_z").get<double>()}}.value();
}

template <>
[[nodiscard]] auto Extrinsics::ToRobotToCamera<frc::Transform3d>() const
    -> frc::Transform3d {
  frc::Pose3d camera_pose(
      units::meter_t{translation_x}, units::meter_t{translation_y},
      units::meter_t{translation_z},
      frc::Rotation3d(units::radian_t{rotation_x}, units::radian_t{rotation_y},
                      units::radian_t{rotation_z}));
  frc::Transform3d robot_to_camera(frc::Pose3d{}, camera_pose);
  return robot_to_camera;
}

template <>
[[nodiscard]] auto Extrinsics::ToCameraToRobot<frc::Transform3d>() const
    -> frc::Transform3d {
  return ToRobotToCamera<frc::Transform3d>().Inverse();
}

template <>
[[nodiscard]] auto Extrinsics::ToCameraToRobot<cv::Mat>() const -> cv::Mat {
  auto camera_to_robot = ToRobotToCamera<frc::Transform3d>().Inverse();
  return utils::EigenToCvMat(camera_to_robot.ToMatrix());
}

}  // namespace camera
