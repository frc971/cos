#pragma once
#include <ostream>
#include <vector>
#include <wpi/math/geometry/Pose3d.hpp>

namespace localization {

using position_estimate_t = struct PositionEstimate {
  std::vector<int> tag_ids;
  std::vector<int> rejected_tag_ids;
  wpi::math::Pose3d pose;
  double variance;
  int num_tags;
  double avg_tag_dist;
  bool invalid = false;
  double loss = 0;

  friend auto operator<<(std::ostream& os,
                         const PositionEstimate& t) -> std::ostream& {
    const auto& tr = t.pose.Translation();
    const auto& r = t.pose.Rotation();
    os << "pose(x=" << tr.X().value() << " y=" << tr.Y().value()
       << " z=" << tr.Z().value() << " roll=" << r.X().value()
       << " pitch=" << r.Y().value() << " yaw=" << r.Z().value() << ")"
       << " variance=" << t.variance << " num_tags=" << t.num_tags;
    return os;
  }
};

}  // namespace localization
