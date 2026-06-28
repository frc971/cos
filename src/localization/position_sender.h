#pragma once

#include <memory>

#include "control_loops/context.h"
#include "localization/position.h"

namespace localization {

class IPositionSender {
 public:
  virtual void Send(const position_estimate_t& estimate,
                    control_loops::MetaDataList metadata,
                    std::shared_ptr<control_loops::Context> ctx) = 0;
  virtual ~IPositionSender() = default;
};

}  // namespace localization
