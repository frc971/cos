#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core/matx.hpp>
#include <frc/geometry/Transform3d.h>

namespace utils {

auto CameraMatrixFromJson(const nlohmann::json& intrinsics) -> cv::Matx33d;
auto DistortionCoefficientsFromJson(const nlohmann::json& intrinsics)
    -> cv::Vec<double, 5>;
// JSON rotations are in degrees; the returned transform uses radians.
auto ExtrinsicsJsonToCameraToRobot(const nlohmann::json& extrinsics)
    -> frc::Transform3d;

}  // namespace utils
