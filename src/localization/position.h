#pragma once

#include <ostream>
#include <typeinfo>
#include <utility>
#include <vector>

#include <frc/geometry/Pose3d.h>

#include "control_loop/message.h"
#include "logging/log_registration.h"

namespace localization {

struct PositionEstimateMessage final : public control_loop::IMessage {
  std::vector<int> tag_ids;
  int num_tags = 0;
  frc::Pose3d pose;
  std::vector<double> distances;
  double variance = 0.0;
  // Mean camera capture time in seconds on the RIO clock (zero if unavailable).
  double timestamp = 0.0;

  PositionEstimateMessage() = default;
  auto GetType() -> const std::type_info& override {
    return typeid(PositionEstimateMessage);
  }
  auto GetSize() -> std::size_t override {
    return sizeof(*this) + tag_ids.capacity() * sizeof(int) +
           distances.capacity() * sizeof(double);
  }

  LOG_FIELDS(PositionEstimateMessage, tag_ids, num_tags, pose, distances,
             variance, timestamp)

  friend auto operator<<(std::ostream& os,
                         const PositionEstimateMessage& estimate)
      -> std::ostream& {
    const auto& translation = estimate.pose.Translation();
    const auto& rotation = estimate.pose.Rotation();
    os << "pose(x=" << translation.X().value()
       << " y=" << translation.Y().value() << " z=" << translation.Z().value()
       << " roll=" << rotation.X().value() << " pitch=" << rotation.Y().value()
       << " yaw=" << rotation.Z().value() << ")"
       << " distances=[";
    for (std::size_t i = 0; i < estimate.distances.size(); ++i) {
      if (i != 0) {
        os << ',';
      }
      os << estimate.distances[i];
    }
    os << "] tag_ids=[";
    for (std::size_t i = 0; i < estimate.tag_ids.size(); ++i) {
      if (i != 0) {
        os << ',';
      }
      os << estimate.tag_ids[i];
    }
    return os << "] variance=" << estimate.variance;
  }
};

using position_estimate_t = PositionEstimateMessage;

}  // namespace localization
