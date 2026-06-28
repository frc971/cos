#pragma once

#include <optional>

#include "localization/position_sender.h"

namespace localization {

class SimSender : public IPositionSender {
 public:
  void Send(const position_estimate_t& estimate) override {
    last_estimate = estimate;
  }

  std::optional<position_estimate_t> last_estimate;
};

}  // namespace localization
