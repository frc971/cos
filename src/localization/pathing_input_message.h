#pragma once

#include <cstddef>
#include <cstdint>

#include <frc/geometry/Pose2d.h>

#include "control_loop/message.h"
#include "pathing/pathing_status.h"

namespace localization {

class PathingInputMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(PathingInputMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
  pathing::PathingStatus status = pathing::PathingStatus::Inactive;
  frc::Pose2d pose;
  frc::Pose2d goal;
  std::int64_t pose_local_us = 0;
  std::int64_t goal_local_us = 0;
  std::int64_t pose_server_us = 0;
  std::int64_t goal_server_us = 0;
};

}  // namespace localization
