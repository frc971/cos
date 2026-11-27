#pragma once

#include <cstddef>
#include <memory>

#include "control_loop/message.h"
#include "pathing/pathing_status.h"

namespace localization {
class PathingInputMessage;
}  // namespace localization

namespace pathing {

// The field-relative command requested by the path follower.
class CommandMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(CommandMessage);
  }
  auto GetSize() -> std::size_t override { return sizeof(*this); }
  double vx = 0;
  double vy = 0;
  double omega = 0;
  PathingStatus status = PathingStatus::Inactive;
  std::shared_ptr<const localization::PathingInputMessage> input;
};

}  // namespace pathing
