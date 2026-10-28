#include "gamepiece/ball_pathing_logic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

#include <opencv2/core.hpp>

namespace gamepiece::internal {
namespace {
auto Distance(const cv::Point2d& first, const cv::Point2d& second) -> double {
  return cv::norm(first - second);
}

auto Better(const BallCluster& first, const BallCluster& second) -> bool {
  return first.evidence_m2 > second.evidence_m2 ||
         (first.evidence_m2 == second.evidence_m2 &&
          cv::norm(first.centroid) < cv::norm(second.centroid));
}

struct GridCell {
  cv::Point2d weighted_position;
  double evidence_m2 = 0.0;
};

auto GridKey(int x, int y) -> std::int64_t {
  const std::uint64_t high = static_cast<std::uint32_t>(x);
  return static_cast<std::int64_t>(
      (high << 32) | static_cast<std::uint32_t>(y));
}
}  // namespace

auto BuildBallClusters(const std::vector<ProjectedMaskSample>& samples,
                       double radius_m, double grid_resolution_m)
    -> std::vector<BallCluster> {
  std::unordered_map<std::int64_t, GridCell> grid;
  for (const ProjectedMaskSample& sample : samples) {
    if (!(sample.evidence_m2 > 0.0) || !std::isfinite(sample.evidence_m2)) {
      continue;
    }
    const int x = static_cast<int>(std::floor(sample.position.x /
                                               grid_resolution_m));
    const int y = static_cast<int>(std::floor(sample.position.y /
                                               grid_resolution_m));
    GridCell& cell = grid[GridKey(x, y)];
    cell.weighted_position += sample.position * sample.evidence_m2;
    cell.evidence_m2 += sample.evidence_m2;
  }
  std::vector<BallCluster> result;
  result.reserve(grid.size());
  for (const auto& [key, center_cell] : grid) {
    static_cast<void>(key);
    const cv::Point2d center =
        center_cell.weighted_position * (1.0 / center_cell.evidence_m2);
    cv::Point2d weighted_position;
    double evidence = 0.0;
    for (const auto& [other_key, cell] : grid) {
      static_cast<void>(other_key);
      const cv::Point2d cell_position =
          cell.weighted_position * (1.0 / cell.evidence_m2);
      if (Distance(center, cell_position) <= radius_m) {
        weighted_position += cell.weighted_position;
        evidence += cell.evidence_m2;
      }
    }
    result.push_back(
        {.centroid = weighted_position * (1.0 / evidence),
         .evidence_m2 = evidence});
  }
  return result;
}

auto ChooseBallCluster(const std::vector<BallCluster>& clusters,
                       const std::optional<BallCluster>& tracked,
                       double match_radius_m, double switch_evidence_ratio)
    -> std::optional<BallCluster> {
  if (clusters.empty()) return std::nullopt;
  const BallCluster& best = *std::max_element(
      clusters.begin(), clusters.end(),
      [](const BallCluster& first, const BallCluster& second) {
        return Better(second, first);
      });
  if (!tracked.has_value()) return best;

  const BallCluster* matched = nullptr;
  for (const BallCluster& cluster : clusters) {
    if (Distance(cluster.centroid, tracked->centroid) <= match_radius_m &&
        (matched == nullptr || Better(cluster, *matched))) {
      matched = &cluster;
    }
  }
  if (matched != nullptr &&
      best.evidence_m2 < matched->evidence_m2 * switch_evidence_ratio) {
    return *matched;
  }
  return best;
}

auto ComputeVelocity(const cv::Point2d& target, double gain_per_s,
                     double maximum_speed_mps, double stop_radius_m)
    -> cv::Point2d {
  const double distance = cv::norm(target);
  if (distance <= stop_radius_m) return {};
  cv::Point2d velocity = target * gain_per_s;
  const double speed = cv::norm(velocity);
  if (speed > maximum_speed_mps) {
    velocity *= maximum_speed_mps / speed;
  }
  return velocity;
}

}  // namespace gamepiece::internal
