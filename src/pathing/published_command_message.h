#pragma once

#include <cstddef>

#include "control_loop/message.h"
#include "pathing/pathing_status.h"

namespace pathing {

// The final field-relative command sent to NetworkTables.
class PublishedCommandMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(PublishedCommandMessage);
  }
  auto GetSize() -> std::size_t override { return sizeof(*this); }

  double vx = 0;
  double vy = 0;
  double omega = 0;
  PathingStatus status = PathingStatus::Inactive;
};

}  // namespace pathing
