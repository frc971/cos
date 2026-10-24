#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Translation2d.h>
#include <networktables/StructTopic.h>
#include <opencv2/core.hpp>

#include "control_loop/node.h"
#include "utils/disjoint_set_union.h"

namespace gamepiece {
struct Ellipse {
  cv::Point2d center;
  cv::Size2d semi_axes;
  double rotation = 0.0;  // radians
};

using kmeans_cluster_t = struct KMeansCluster {
  cv::Point2f
      centroid;  // Focal-length-normalized image coordinates (x/z, y/z).
  cv::Matx22f covar{};  // Covariance of img_points in normalized coordinates.
  std::vector<cv::Point2f> img_points;
  std::optional<frc::Translation2d> robot_relative_translation = std::nullopt;
  std::vector<cv::Point2f> ground_points;  // Robot coordinates, meters.
  std::vector<float>
      ground_weights;  // Visible silhouette support in Fuel equivalents.
  cv::Matx22f ground_covar{};
  float fuel_count{};
  float density{};  // Estimated visible balls / physical Gaussian area.
  float estimated_ball_count{};
  float
      density_area_m2{};  // Effective Gaussian footprint, with a ball-radius floor.
  std::size_t verified_ball_count{};
  std::vector<cv::Point2f> verified_ball_centers;  // Robot coordinates, meters.
  cv::Vec2f
      residual_ball_evidence{};  // Unexplained area / sphere area, arc / perimeter.
  cv::Vec2f
      total_ball_evidence{};  // Geometry-normalized area and edge evidence.
  float
      instantaneous_ball_count{};  // Current evidence before the temporal update.
  float instantaneous_density_area_m2{};
  std::size_t predicted_ball_count{}, reused_ball_count{};
  std::size_t id{};  // Persistent within one tracker instance.
};

using hsv_clusters_t = std::vector<kmeans_cluster_t>;
enum class DensityHistory { kNone, kFilter, kPredict };

// The normalized 2-sigma ellipse for image annotation.
[[nodiscard]] auto ClusterEllipse(const kmeans_cluster_t& cluster) -> Ellipse;

class HSVClusterTrackerNode final : public control_loop::INode {
 public:
  HSVClusterTrackerNode(
      std::string_view input_path, std::string_view output_path,
      cv::Matx33d intrinsics, cv::Matx44d extrinsics,
      std::size_t image_size,  // Total pixels (width * height).
      std::string_view pose_path = "",
      DensityHistory density_history = DensityHistory::kPredict,
      bool residual_density = true,
      std::string_view pose_topic = "/Robot/Pose",
      nt::NetworkTableInstance instance = nt::NetworkTableInstance::GetDefault());
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
  auto ClusterGroundPoints(const std::vector<cv::Point2f>& image_points,
                           const std::vector<cv::Point2f>& ground_points,
                           const std::vector<float>& ground_weights,
                           const std::vector<float>& refined_weights,
                           const std::optional<frc::Pose2d>& robot_pose) const
      -> hsv_clusters_t;
  void AssignPersistentIds(hsv_clusters_t& clusters,
                           const std::optional<frc::Pose2d>& robot_pose);
  void EstimateDensities(const cv::Mat& frame, hsv_clusters_t& clusters,
                         const std::optional<frc::Pose2d>& robot_pose);
  static auto SphereArcWeight(const cv::Matx33f& intrinsics,
                              const cv::Vec3f& ray, const cv::Vec2f& normal,
                              float depth) -> float;
  struct BallTrack {
    cv::Point2f position;  // Field coordinates when pose is available.
    std::size_t observations = 1, missed = 0;
  };
  struct DensityState {
    float count = 0, innovation = 0;
    cv::Matx22f covariance{};
    std::size_t updates = 0, missed = 0, changed = 0;
  };
  std::vector<BallTrack> ball_tracks_;
  std::vector<DensityState> density_states_;
  bool temporal_density_;
  bool predictive_density_;
  bool residual_density_;

  std::string input_path_;
  std::string output_path_;
  std::string pose_path_;
  nt::StructSubscriber<frc::Pose3d> pose_subscriber_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex processing_mutex_;
  // Retained for callbacks without a new pose and coordinate-mode changes.
  std::optional<frc::Pose2d> last_robot_pose_;
  std::vector<kmeans_cluster_t> clusters_;
  // Indexed by ID; field positions when a robot pose is provided.
  // Retain positions through empty frames for re-entry matching.
  std::vector<frc::Translation2d> canonical_positions_;
  std::vector<cv::Matx22f> canonical_covariances_;

  cv::Mat3b hsv_image_;
  cv::Mat1b hsv_masked_;
  cv::Matx33f camera_intrinsics_;
  cv::Matx44f camera_extrinsics_cv_;
  cv::Vec3f camera_position_;
  static constexpr float kIdMatchToleranceM = 0.9f;
  cv::Mat frame_;
  frc::Translation2d extrinsics_offset_;
};

}  // namespace gamepiece
