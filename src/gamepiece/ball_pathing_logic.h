#pragma once

#include <optional>
#include <opencv2/core/types.hpp>
#include <vector>

namespace gamepiece::internal {

struct BallCluster {
  cv::Point2d centroid;
  double evidence_m2 = 0.0;
};

struct ProjectedMaskSample {
  cv::Point2d position;
  double evidence_m2 = 0.0;
};

auto BuildBallClusters(const std::vector<ProjectedMaskSample>& samples,
                       double radius_m, double grid_resolution_m)
    -> std::vector<BallCluster>;

auto ChooseBallCluster(const std::vector<BallCluster>& clusters,
                       const std::optional<BallCluster>& tracked,
                       double match_radius_m, double switch_evidence_ratio)
    -> std::optional<BallCluster>;

auto ComputeVelocity(const cv::Point2d& target, double gain_per_s,
                     double maximum_speed_mps, double stop_radius_m)
    -> cv::Point2d;

}  // namespace gamepiece::internal
