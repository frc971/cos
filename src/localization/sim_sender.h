#pragma once

#include <optional>

#include "localization/position_sender.h"

namespace localization {

class SimSender : public IPositionSender {
 public:
  void Send(const position_estimate_t& estimate,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context> ctx) override {
    last_estimate = estimate;
    last_metadata = std::move(metadata);
    last_context = std::move(ctx);
  }

  std::optional<position_estimate_t> last_estimate;
  control_loops::MetaDataList last_metadata;
  std::shared_ptr<control_loops::Context> last_context;
};

}  // namespace localization
