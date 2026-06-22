#pragma once

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <wpi/math/geometry/Transform3d.hpp>
#include "third_party/971apriltag/apriltag.h"

namespace utils {

template <typename T>
auto CameraMatrixFromJson(nlohmann::json intrinsics) -> T;

template <typename T>
auto DistortionCoefficientsFromJson(nlohmann::json intrinsics) -> T;

auto ExtrinsicsJsonToCameraToRobot(nlohmann::json extrinsics_json)
    -> wpi::math::Transform3d;

}  // namespace utils
