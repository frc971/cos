#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <frc/geometry/Pose2d.h>
#include <opencv2/core.hpp>

#include "control_loop/node.h"
#include "gamepiece/ellipse.h"
#include "utils/disjoint_set_union.h"

namespace gamepiece {

using kmeans_cluster_t = struct KMeansCluster {
  cv::Point2f
      centroid;  // Undistorted, focal-length-normalized image coordinates (x/z, y/z).
  cv::Matx22f covar{};  // Covariance of img_points in normalized coordinates.
  std::vector<cv::Point2f> img_points;
  std::optional<frc::Translation2d> robot_relative_translation = std::nullopt;
  std::vector<cv::Point2f> ground_points;  // Robot coordinates, meters.
  std::vector<float> ground_weights;  // Visible silhouette support in Fuel equivalents.
  cv::Matx22f ground_covar{};
  float fuel_count{};
  std::size_t id{};  // Persistent within one tracker instance.
};

using hsv_clusters_t = std::vector<kmeans_cluster_t>;

// The normalized 2-sigma ellipse for image annotation.
[[nodiscard]] auto ClusterEllipse(const kmeans_cluster_t& cluster) -> Ellipse;

class HSVClusterTrackerNode final : public control_loop::INode {
 public:
  HSVClusterTrackerNode(
      std::string_view input_path, std::string_view output_path,
      cv::Matx33d intrinsics, cv::Matx44d extrinsics,
      const cv::Vec<double, 5>& distortion_coeffs,
      std::size_t image_size,  // Total pixels (width * height).
      float max_ground_distance_m = 20.0f, std::string_view pose_path = "");
  // Processes a CPU BGR frame; callers must serialize access with callbacks.
  void ProcessFrame(const cv::Mat& frame,
                    std::optional<frc::Pose2d> robot_pose = std::nullopt);
  [[nodiscard]] auto GetClusters() const
      -> const std::vector<kmeans_cluster_t>*;
  // Generic ray intersection with the Fuel center plane. Silhouette-corrected
  // retained samples are in each cluster's ground_points.
  [[nodiscard]] auto RobotRelativeOffset(const cv::Point2f& point) const
      -> frc::Translation2d;

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  friend struct HSVClusterTrackerNodeTestPeer;
  [[nodiscard]] auto NormalizedPointOffset(const cv::Point2f& point) const
      -> frc::Translation2d;
  auto ClusterGroundPoints() -> std::vector<kmeans_cluster_t>;
  void AssignPersistentIds();

  std::string input_path_;
  std::string output_path_;
  std::string pose_path_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex processing_mutex_;
  std::vector<cv::Point2f> thresholded_points_;
  std::vector<cv::Point2f> ground_points_;
  std::vector<float> ground_weights_;
  std::vector<float> refined_weights_;
  std::optional<frc::Pose2d> robot_pose_;
  std::vector<kmeans_cluster_t> clusters_;
  // Indexed by ID; field positions when a robot pose is provided.
  // Retain positions through empty frames for re-entry matching.
  std::vector<frc::Translation2d> canonical_positions_;
  std::vector<cv::Matx22f> canonical_covariances_;

  cv::Mat3b hsv_image_;
  cv::Mat1b hsv_masked_;
  cv::Matx33f camera_intrinsics_;
  cv::Vec<float, 5> distortion_coeffs_;
  cv::Matx44f camera_extrinsics_cv_;
  float max_ground_distance_m_;
  static constexpr float kIdMatchToleranceM = 0.9f;
  cv::Mat frame_;
  frc::Translation2d extrinsics_offset_;
};

}  // namespace gamepiece
