#pragma once

#include <wpi/math/geometry/Pose3d.hpp>

namespace gamepiece {

struct gamepiece_detection_t {
  wpi::math::Pose3d pose;
  int tracker_id = -1;
  int class_id = -1;
  float confidence = 0.0F;
};

}  // namespace gamepiece
