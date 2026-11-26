#pragma once

#include <frc/geometry/Pose2d.h>

#include "control_loop/message.h"

namespace pathing {

// A field-relative pose used on either the measured-pose or goal channel.
class PoseMessage final : public control_loop::IMessage {
 public:
  explicit PoseMessage(frc::Pose2d pose) : goal_position(pose) {}
  auto GetType() -> const std::type_info& override {
    return typeid(PoseMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
  frc::Pose2d goal_position;
};

}  // namespace pathing
