#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include "utils/disjoint_set_union.h"

#include <frc/geometry/Translation2d.h>

namespace gamepiece {

using kmeans_cluster_t = struct KMeansCluster {
  cv::Point2f centroid;
  cv::Mat covar;
  std::vector<cv::Point2f> img_points;
  std::optional<frc::Translation2d> camera_relative_translation = std::nullopt;
};

class HSVClusterTracker {
 public:
  HSVClusterTracker(cv::Matx33d intrinsics, cv::Matx44d extrinsics,
                    const cv::Vec<double, 5>& distortion_coeffs,
                    float max_ground_distance_m = 20.0f);
  void ProcessFrame(const cv::Mat& frame);
  [[nodiscard]] auto GetClusters() const
      -> const std::vector<kmeans_cluster_t>*;

 private:
  [[nodiscard]] auto UndistortedPixelPointOffset(const cv::Point2f& point) const
      -> std::optional<frc::Translation2d>;
  [[nodiscard]] auto KMeans(
      const std::vector<cv::Point2f>& data_points, int k,
      const std::vector<kmeans_cluster_t>& initial_clusters) const
      -> std::vector<kmeans_cluster_t>;
  [[nodiscard]] auto AssignToExistingClusters(
      const std::vector<cv::Point2f>& data_points,
      const std::vector<kmeans_cluster_t>& existing_clusters) const
      -> std::vector<kmeans_cluster_t>;
  [[nodiscard]] auto CovarianceSpikeCount(
      const std::vector<kmeans_cluster_t>& previous_clusters,
      const std::vector<kmeans_cluster_t>& assigned_clusters) const -> int;
  [[nodiscard]] auto ClusterDistance(const kmeans_cluster_t& cluster) const
      -> frc::Translation2d;
  [[nodiscard]] auto ClustersOverlap(const kmeans_cluster_t& first,
                                     const kmeans_cluster_t& second) const
      -> bool;
  auto MergeOverlappingClusters(
      const std::vector<kmeans_cluster_t>& unfiltered_clusters)
      -> std::vector<kmeans_cluster_t>;

  int active_cluster_count_ = 20;
  std::vector<cv::Point2f> thresholded_points_;
  std::vector<kmeans_cluster_t> clusters_;
  utils::DisjointSetUnion cluster_dsu_;
  cv::Mat3b hsv_image_;
  cv::Mat1b hsv_masked_;
  cv::Matx33d camera_intrinsics_;
  cv::Vec<double, 5> distortion_coeffs_;
  cv::Matx44f camera_extrinsics_cv_;
  float max_ground_distance_m_;
  static constexpr float max_merge_distance_m{0.5f};
};

}  // namespace gamepiece
