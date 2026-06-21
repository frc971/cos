#pragma once

#include <nlohmann/json.hpp>
#include "third_party/971apriltag/apriltag.h"

namespace utils {

template <typename T>
auto CameraMatrixFromJson(nlohmann::json intrinsics) -> T;

template <typename T>
auto DistortionCoefficientsFromJson(nlohmann::json intrinsics) -> T;

}  // namespace utils
