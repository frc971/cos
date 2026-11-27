#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "control_loop/message.h"
#include "pathing/pathing_status.h"

namespace pathing {

// The field-relative command requested by the path follower.
class CommandMessage final : public control_loop::IMessage {
 public:
  struct InputTimestamps {
    std::int64_t pose_server_us = 0;
    std::int64_t goal_server_us = 0;
  };

  auto GetType() -> const std::type_info& override {
    return typeid(CommandMessage);
  }
  auto GetSize() -> std::size_t override { return sizeof(*this); }
  double vx = 0;
  double vy = 0;
  double omega = 0;
  PathingStatus status = PathingStatus::Inactive;
  // The NetworkTables packet carries the original pose and goal sample times
  // in server microseconds so the robot can identify the inputs used for this
  // command. Commands created without an input snapshot, such as the shutdown
  // stop command, leave this empty.
  std::optional<InputTimestamps> input_timestamps;
};

}  // namespace pathing
