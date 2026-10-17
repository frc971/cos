#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <opencv2/core.hpp>

#include "control_loop/node.h"
#include "utils/disjoint_set_union.h"

namespace gamepiece {

using kmeans_cluster_t = struct KMeansCluster {
  // Undistorted, focal-length-normalized image coordinates (x/z, y/z).
  cv::Point2f centroid;
  // Covariance of img_points in normalized coordinates.
  cv::Mat covar;
  std::vector<cv::Point2f> img_points;
  std::optional<frc::Translation2d> camera_relative_translation = std::nullopt;
};

using hsv_clusters_t = std::vector<kmeans_cluster_t>;

class HSVClusterTrackerNode final : public control_loop::INode {
 public:
  HSVClusterTrackerNode(std::string_view input_path,
                        std::string_view output_path, cv::Matx33d intrinsics,
                        cv::Matx44d extrinsics,
                        const cv::Vec<double, 5>& distortion_coeffs,
                        float max_ground_distance_m = 20.0f);
  // Processes a CPU BGR frame; callers must serialize access with callbacks.
  void ProcessFrame(const cv::Mat& frame);
  [[nodiscard]] auto GetClusters() const
      -> const std::vector<kmeans_cluster_t>*;

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  [[nodiscard]] auto NormalizedPointOffset(const cv::Point2f& point) const
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

  std::string input_path_;
  std::string output_path_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex processing_mutex_;
  int active_cluster_count_ = 20;
  std::vector<cv::Point2f> thresholded_points_;
  std::vector<kmeans_cluster_t> clusters_;
  utils::DisjointSetUnion cluster_dsu_;
  cv::Mat3b hsv_image_;
  cv::Mat1b hsv_masked_;
  cv::Matx33f camera_intrinsics_;
  cv::Vec<float, 5> distortion_coeffs_;
  cv::Matx44f camera_extrinsics_cv_;
  float max_ground_distance_m_;
  static constexpr float max_merge_distance_m{0.5f};
  cv::Mat frame_;
};

}  // namespace gamepiece
