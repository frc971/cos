#pragma once

#include <cstddef>
#include <vector>

#include <frc/geometry/Pose3d.h>

#include "control_loop/message.h"

namespace gamepiece {

struct gamepiece_detection_t {
  frc::Pose3d pose;
  int tracker_id = -1;
  int class_id = -1;
  float confidence = 0.0F;
};

class GamepieceDetections final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(GamepieceDetections);
  }
  auto GetSize() -> size_t override {
    return sizeof(*this) +
           detections.capacity() * sizeof(gamepiece_detection_t);
  }

  std::vector<gamepiece_detection_t> detections;
};

}  // namespace gamepiece
