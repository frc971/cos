#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>

#include "gamepiece/cluster_nt.h"
#include "gamepiece/hsv_cluster_tracker_node.h"

namespace streamer::detail {

inline auto FieldCluster(const gamepiece::kmeans_cluster_t& cluster,
                         const frc::Pose2d& pose)
    -> std::optional<gamepiece::cluster_nt_t> {
  if (!cluster.robot_relative_translation)
    return std::nullopt;
  const auto center =
      cluster.robot_relative_translation->RotateBy(pose.Rotation()) +
      pose.Translation();
  const double heading = pose.Rotation().Radians().value();
  const double xx = cluster.ground_covar(0, 0);
  const double yy = cluster.ground_covar(1, 1);
  const double xy =
      0.5 * (cluster.ground_covar(0, 1) + cluster.ground_covar(1, 0));
  const double radius = std::hypot(xx - yy, 2 * xy);
  const double major_variance = 0.5 * (xx + yy + radius);
  const double minor_variance = 0.5 * (xx + yy - radius);
  if (!std::isfinite(center.X().value()) ||
      !std::isfinite(center.Y().value()) || !std::isfinite(heading) ||
      !std::isfinite(major_variance) || !std::isfinite(minor_variance) ||
      minor_variance < -1e-6 * std::max(1.0, major_variance) ||
      !std::isfinite(cluster.density) || cluster.density < 0) {
    return std::nullopt;
  }
  double yaw =
      std::fmod(0.5 * std::atan2(2 * xy, xx - yy) + heading, std::numbers::pi);
  if (yaw < 0)
    yaw += std::numbers::pi;
  return gamepiece::cluster_nt_t{
      .center_x = center.X().value(),
      .center_y = center.Y().value(),
      .field_relative_semimajor_axis_yaw = yaw,
      .semimajor_length = 2 * std::sqrt(std::max(0.0, major_variance)),
      .semiminor_length = 2 * std::sqrt(std::max(0.0, minor_variance)),
      .predicted_density = cluster.density};
}

}  // namespace streamer::detail
