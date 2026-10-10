#pragma once
#include <frc/geometry/Transform3d.h>
#include <filesystem>
#include <opencv2/core/mat.hpp>
namespace camera {

struct Intrinsics {
  Intrinsics(const std::filesystem::path& path);
  [[nodiscard]] auto ToMatrix() const -> cv::Mat;
  [[nodiscard]] auto ToDistortionCoefficients() const -> cv::Mat;

  int width;
  int height;
  double cx;
  double cy;
  double fx;
  double fy;

  double k1 = 0;
  double k2 = 0;
  double k3 = 0;
  double p1 = 0;
  double p2 = 0;
};

struct Extrinsics {
  Extrinsics(const std::filesystem::path& path);

  template <typename T>
  [[nodiscard]] auto ToCameraToRobot() const -> T;

  template <typename T>
  [[nodiscard]] auto ToRobotToCamera() const -> T;

  double translation_x = 0;
  double translation_y = 0;
  double translation_z = 0;
  // Rotations are stored in radians internally and degrees in JSON.
  double rotation_x = 0;
  double rotation_y = 0;
  double rotation_z = 0;
};

template <>
[[nodiscard]] auto Extrinsics::ToCameraToRobot<cv::Mat>() const -> cv::Mat;

template <>
[[nodiscard]] auto Extrinsics::ToCameraToRobot<frc::Transform3d>() const
    -> frc::Transform3d;

template <>
[[nodiscard]] auto Extrinsics::ToRobotToCamera<frc::Transform3d>() const
    -> frc::Transform3d;

}  // namespace camera
