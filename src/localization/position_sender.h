#pragma once

#include "localization/position.h"

namespace localization {

class IPositionSender {
 public:
  virtual void Send(const position_estimate_t& estimate) = 0;
  virtual ~IPositionSender() = default;
};

}  // namespace localization
