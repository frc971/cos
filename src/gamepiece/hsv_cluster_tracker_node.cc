#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "camera/nvjpeg_decode_node.h"
#include "gamepiece/ellipse.h"
#include "gamepiece/image_utils.h"
#include "utils/cv_geometry.h"

namespace gamepiece {
namespace {

constexpr std::pair<int, int> hsv_color_range{18, 30};
constexpr int minimum_saturation{150};

constexpr int kInitialClusterCount = 10;
constexpr float kCovarianceSpikeRatio = 2.5f;
constexpr double kmin_pixel_ratio = 0.0001;

auto Covariance(const std::vector<cv::Point2f>& points) -> cv::Matx22f {
  cv::Point2f mean{0.0f, 0.0f};
  for (const cv::Point2f& point : points) {
    mean += point;
  }
  mean /= static_cast<float>(points.size());

  cv::Matx22f covariance{};
  for (const cv::Point2f& point : points) {
    const cv::Point2f offset = point - mean;
    covariance(0, 0) += offset.x * offset.x;
    covariance(0, 1) += offset.x * offset.y;
    covariance(1, 0) += offset.y * offset.x;
    covariance(1, 1) += offset.y * offset.y;
  }
  return covariance / static_cast<float>(points.size());
}

}  // namespace

HSVClusterTrackerNode::HSVClusterTrackerNode(
    std::string_view input_path, std::string_view output_path,
    cv::Matx33d intrinsics, cv::Matx44d extrinsics,
    const cv::Vec<double, 5>& distortion_coeffs, std::size_t image_size,
    const float max_ground_distance_m)
    : input_path_(input_path),
      output_path_(output_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_(
          {control_loop::MessageDescriptor::Publication<hsv_clusters_t>(
              output_path_)}),
      camera_intrinsics_(intrinsics),
      distortion_coeffs_(distortion_coeffs),
      camera_extrinsics_cv_(extrinsics),
      max_ground_distance_m_(max_ground_distance_m),
      image_size_(image_size) {
  if (image_size_ == 0) {
    throw std::invalid_argument("Image size must be positive");
  }
  if (!std::isfinite(max_ground_distance_m_) || max_ground_distance_m_ <= 0) {
    throw std::invalid_argument(
        "Maximum ground distance must be finite and positive");
  }
  if (!std::isfinite(extrinsics(2, 3)) || extrinsics(2, 3) <= 0) {
    throw std::invalid_argument(
        "HSV tracker extrinsics must be the camera pose in robot coordinates "
        "with a positive camera height");
  }
  // Capture the ground translation before changing from WPILib coordinates.
  extrinsics_offset_ = frc::Translation2d{units::meter_t{extrinsics(0, 3)},
                                          units::meter_t{extrinsics(1, 3)}};
  utils::ChangeBasis(camera_extrinsics_cv_, utils::Basis::kWpiToCv);
}

auto HSVClusterTrackerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    std::unique_lock processing_lock(processing_mutex_);
    if (context->Exists(output_path_)) {
      return;
    }
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      ProcessFrame(cv::Mat{});
      context->SetMessage(output_path_, nullptr);
    } else {
      CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_BGRI);
      CHECK_GE(frame->stride, static_cast<size_t>(frame->width) * 3U);
      const cv::cuda::GpuMat image(frame->height, frame->width, CV_8UC3,
                                   frame->destination.channel[0],
                                   frame->stride);
      image.download(frame_);
      ProcessFrame(frame_);
      context->SetMessage(
          output_path_,
          std::make_shared<control_loop::ValueMessage<hsv_clusters_t>>(
              clusters_));
    }
    processing_lock.unlock();
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

void HSVClusterTrackerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto HSVClusterTrackerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto HSVClusterTrackerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto HSVClusterTrackerNode::NormalizedPointOffset(
    const cv::Point2f& point) const -> std::optional<frc::Translation2d> {
  const auto offset =
      UndistortedPinholePointOffset(point, 0, camera_extrinsics_cv_);
  if (!offset.has_value() || !std::isfinite(offset->Norm().value()) ||
      offset->Norm().value() > max_ground_distance_m_) {
    return std::nullopt;
  }
  return offset;
}

auto HSVClusterTrackerNode::RobotRelativeOffset(const cv::Point2f& point) const
    -> std::optional<frc::Translation2d> {
  const auto offset = NormalizedPointOffset(point);
  return offset.has_value() ? std::optional{*offset + extrinsics_offset_}
                            : std::nullopt;
}

void HSVClusterTrackerNode::ProcessFrame(const cv::Mat& frame) {
  const std::vector<kmeans_cluster_t> previous_clusters = clusters_;
  clusters_.clear();
  thresholded_points_.clear();
  bottom_columns_.clear();

  if (frame.empty()) {
    return;
  }

  HSVThreshold(frame, hsv_color_range.first, hsv_color_range.second,
               minimum_saturation, thresholded_points_, hsv_image_,
               hsv_masked_);
  if (thresholded_points_.empty()) {
    return;
  }
  cv::undistortPoints(thresholded_points_, thresholded_points_,
                      camera_intrinsics_, distortion_coeffs_);
  std::erase_if(thresholded_points_, [this](const cv::Point2f& point) {
    return !NormalizedPointOffset(point).has_value();
  });
  if (thresholded_points_.empty()) {
    return;
  }
  bottom_columns_ = LowestPointsByColumn(
      thresholded_points_, frame.cols,
      static_cast<double>(camera_intrinsics_(0, 0)));
  clusters_ = AssignPersistentIds(
      MergeOverlappingClusters(KMeans(thresholded_points_, previous_clusters)));
}

auto HSVClusterTrackerNode::ClampToHorizon(cv::Point2f point) const
    -> cv::Point2f {
  const float y_coefficient = camera_extrinsics_cv_(1, 1);
  const float horizon_y =
      -(camera_extrinsics_cv_(1, 0) * point.x + camera_extrinsics_cv_(1, 2)) /
      y_coefficient;
  // Stay just on the ground side to keep the floor projection finite.
  const float ground_y = horizon_y + std::copysign(1e-4f, y_coefficient);
  point.y = y_coefficient > 0 ? std::max(point.y, ground_y)
                              : std::min(point.y, ground_y);
  return point;
}

auto HSVClusterTrackerNode::GroundPoint(const cv::Point2f& point,
                                        bool use_column_bottom) const
    -> cv::Point2f {
  std::optional<frc::Translation2d> projected_offset;
  if (use_column_bottom && !bottom_columns_.empty()) {
    projected_offset = NormalizedPointOffset(
        {point.x, InterpolatedLowestY(bottom_columns_, point.x)});
  }
  // Original rays are filtered frame points or horizon-clamped cluster seeds.
  const auto offset = projected_offset.has_value()
                          ? *projected_offset
                          : UndistortedPinholePointOffset(
                                point, 0, camera_extrinsics_cv_).value();
  return {static_cast<float>(offset.X().value()),
          static_cast<float>(offset.Y().value())};
}

auto HSVClusterTrackerNode::AssignToExistingClusters(
    const std::vector<cv::Point2f>& data_points,
    const std::vector<kmeans_cluster_t>& existing_clusters) const
    -> std::vector<kmeans_cluster_t> {
  if (existing_clusters.empty()) {
    return {};
  }
  std::vector<kmeans_cluster_t> assigned_clusters(existing_clusters.size());
  std::vector<cv::Point2f> scaled_centroids;
  scaled_centroids.reserve(existing_clusters.size());
  for (std::size_t i = 0; i < existing_clusters.size(); ++i) {
    assigned_clusters[i].centroid =
        ClampToHorizon(existing_clusters[i].centroid);
    scaled_centroids.push_back(
        GroundPoint(assigned_clusters[i].centroid, false));
  }
  for (const auto& point : data_points) {
    const cv::Point2f scaled_point = GroundPoint(point);
    const auto nearest = std::ranges::min_element(
        scaled_centroids, {}, [&](const cv::Point2f& center) {
          return cv::norm(scaled_point - center);
        });
    assigned_clusters[nearest - scaled_centroids.begin()].img_points.push_back(
        point);
  }
  for (auto& cluster : assigned_clusters) {
    if (!cluster.img_points.empty()) {
      cluster.covar = Covariance(cluster.img_points);
    }
  }
  return assigned_clusters;
}

auto HSVClusterTrackerNode::KMeans(
    const std::vector<cv::Point2f>& data_points,
    const std::vector<kmeans_cluster_t>& initial_clusters) const
    -> std::vector<kmeans_cluster_t> {
  if (data_points.empty()) {
    return {};
  }
  auto seeds = initial_clusters;
  if (seeds.empty()) {
    const int initial_k =
        std::min(kInitialClusterCount, static_cast<int>(data_points.size()));
    seeds.resize(initial_k);
    for (int i = 0; i < initial_k; ++i) {
      seeds[i].centroid = data_points[i * data_points.size() / initial_k];
    }
  }
  auto assigned_clusters = AssignToExistingClusters(data_points, seeds);
  for (std::size_t i = 0; i < initial_clusters.size(); ++i) {
    const auto& assigned = assigned_clusters[i];
    const float previous_spread = cv::trace(initial_clusters[i].covar);
    const float assigned_spread = cv::trace(assigned.covar);
    if (assigned.img_points.empty() ||
        previous_spread <= std::numeric_limits<float>::epsilon() ||
        assigned_spread <= previous_spread * kCovarianceSpikeRatio) {
      continue;
    }
    const Ellipse ellipse = ClusterEllipse(assigned);
    const cv::Point2f offset{
        static_cast<float>(ellipse.semi_axes.width *
                           std::cos(ellipse.rotation) / 3.0),
        static_cast<float>(ellipse.semi_axes.width *
                           std::sin(ellipse.rotation) / 3.0)};
    std::vector<kmeans_cluster_t> split(2);
    split[0].centroid = assigned.centroid - offset;
    split[1].centroid = assigned.centroid + offset;
    split = AssignToExistingClusters(assigned.img_points, split);
    assigned_clusters[i] = std::move(split[0]);
    assigned_clusters.push_back(std::move(split[1]));
  }
  std::erase_if(assigned_clusters, [](const auto& cluster) {
    return cluster.img_points.empty();
  });
  const int k = static_cast<int>(assigned_clusters.size());
  std::vector<cv::Point2f> points, scaled_points;
  std::vector<int> initial_labels;
  points.reserve(data_points.size());
  scaled_points.reserve(data_points.size());
  initial_labels.reserve(data_points.size());
  for (int i = 0; i < k; ++i) {
    for (const auto& point : assigned_clusters[i].img_points) {
      points.push_back(point);
      scaled_points.push_back(GroundPoint(point));
      initial_labels.push_back(i);
    }
  }
  cv::Mat labels(initial_labels, true);
  cv::Mat centers;
  const cv::TermCriteria criteria(
      cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-4);
  cv::kmeans(scaled_points, k, labels, criteria, 1,
             cv::KMEANS_USE_INITIAL_LABELS, centers);

  std::vector<kmeans_cluster_t> clusters(k);
  for (int i = 0; i < labels.rows; ++i) {
    auto& cluster = clusters[labels.at<int>(i)];
    cluster.centroid += points[i];
    cluster.img_points.push_back(points[i]);
  }
  std::erase_if(clusters, [this](const auto& cluster) {
    return cluster.img_points.size() < kmin_pixel_ratio * image_size_;
  });
  for (auto& cluster : clusters) {
    cluster.centroid /= static_cast<float>(cluster.img_points.size());
    cluster.covar = Covariance(cluster.img_points);
  }
  return clusters;
}

auto HSVClusterTrackerNode::ClusterDistance(
    const kmeans_cluster_t& cluster) const -> frc::Translation2d {
  if (cluster.img_points.empty()) {
    return {};
  }

  return NormalizedPointOffset(cluster.centroid).value();
}

auto HSVClusterTrackerNode::ClustersOverlap(
    const kmeans_cluster_t& first, const kmeans_cluster_t& second) const
    -> bool {
  return ellipse_overlap_area(ClusterEllipse(first), ClusterEllipse(second)) >
         0.0f;
}

auto ClusterEllipse(const kmeans_cluster_t& cluster) -> Ellipse {
  const cv::Mat covariance(cluster.covar);
  if (covariance.rows != 2 || covariance.cols != 2) {
    throw std::invalid_argument("KMeans covariance must be 2 by 2");
  }

  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  cv::eigen(covariance, eigenvalues, eigenvectors);
  constexpr float kMinimumRadius = 1e-6f;
  const float major =
      2.0 * std::sqrt(std::max(eigenvalues.at<float>(0, 0), 0.0f)) +
      kMinimumRadius;
  const float minor =
      2.0 * std::sqrt(std::max(eigenvalues.at<float>(1, 0), 0.0f)) +
      kMinimumRadius;
  const cv::Vec2f major_axis(eigenvectors.at<float>(0, 0),
                             eigenvectors.at<float>(0, 1));
  return {.center = {cluster.centroid.x, cluster.centroid.y},
          .semi_axes = {major, minor},
          .rotation = std::atan2(major_axis[1], major_axis[0])};
}

auto HSVClusterTrackerNode::MergeOverlappingClusters(
    const std::vector<kmeans_cluster_t>& unfiltered_clusters)
    -> std::vector<kmeans_cluster_t> {
  cluster_dsu_.Clear();
  cluster_dsu_.FillSets(unfiltered_clusters.size());
  std::vector<frc::Translation2d> cluster_positions;
  cluster_positions.reserve(unfiltered_clusters.size());
  for (const auto& cluster : unfiltered_clusters) {
    cluster_positions.push_back(ClusterDistance(cluster));
  }

  for (std::size_t first = 0; first < unfiltered_clusters.size(); ++first) {
    for (std::size_t second = first + 1; second < unfiltered_clusters.size();
         ++second) {
      if (cluster_positions[first]
                  .Distance(cluster_positions[second])
                  .value() <= max_merge_distance_m &&
          ClustersOverlap(unfiltered_clusters[first],
                          unfiltered_clusters[second])) {
        cluster_dsu_.Union(first, second);
      }
    }
  }

  return MergeClusters(unfiltered_clusters);
}

auto HSVClusterTrackerNode::AssignPersistentIds(
    std::vector<kmeans_cluster_t> clusters)
    -> std::vector<kmeans_cluster_t> {
  cluster_dsu_.Clear();
  cluster_dsu_.FillSets(clusters.size());
  const std::size_t canonical_count = canonical_positions_.size();
  std::vector<std::size_t> first_cluster(canonical_count + clusters.size(),
                                        clusters.size());
  // Match only positions from earlier frames, before updating any canonical.
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    auto& cluster = clusters[i];
    const auto position = ClusterDistance(cluster) + extrinsics_offset_;
    cluster.id = canonical_positions_.size();
    double nearest_distance = std::numeric_limits<double>::infinity();
    for (std::size_t id = 0; id < canonical_count; ++id) {
      const double distance = position.Distance(canonical_positions_[id]).value();
      if (distance <= kIdMatchToleranceM && distance < nearest_distance) {
        cluster.id = id;
        nearest_distance = distance;
      }
    }
    if (cluster.id == canonical_positions_.size()) {
      canonical_positions_.push_back(position);
    }
    auto& first = first_cluster[cluster.id];
    if (first == clusters.size()) {
      first = i;
    } else {
      cluster_dsu_.Union(first, i);
    }
  }
  auto merged = MergeClusters(clusters);
  for (auto& cluster : merged) {
    cluster.robot_relative_translation =
        ClusterDistance(cluster) + extrinsics_offset_;
    canonical_positions_[cluster.id] = *cluster.robot_relative_translation;
  }
  return merged;
}

auto HSVClusterTrackerNode::MergeClusters(
    const std::vector<kmeans_cluster_t>& unfiltered_clusters)
    -> std::vector<kmeans_cluster_t> {
  std::vector<kmeans_cluster_t> merged_clusters;
  merged_clusters.reserve(cluster_dsu_.ComponentCount());
  std::vector<std::size_t> component_cluster(unfiltered_clusters.size(),
                                             unfiltered_clusters.size());

  for (std::size_t i = 0; i < unfiltered_clusters.size(); ++i) {
    const std::size_t component = cluster_dsu_.Find(i);
    std::size_t& merged_index = component_cluster[component];
    if (merged_index == unfiltered_clusters.size()) {
      merged_index = merged_clusters.size();
      merged_clusters.push_back(unfiltered_clusters[i]);
      merged_clusters.back().img_points.clear();
    }

    auto& merged = merged_clusters[merged_index];
    const auto& points = unfiltered_clusters[i].img_points;
    merged.img_points.insert(merged.img_points.end(), points.begin(),
                             points.end());
  }

  for (kmeans_cluster_t& merged : merged_clusters) {
    if (merged.img_points.empty()) {
      continue;
    }

    cv::Point2f point_sum{0.0f, 0.0f};
    for (const cv::Point2f& point : merged.img_points) {
      point_sum += point;
    }
    merged.centroid = point_sum / static_cast<float>(merged.img_points.size());

    if (merged.img_points.size() > 1) {
      merged.covar = Covariance(merged.img_points);
    } else {
      merged.covar = cv::Matx22f::zeros();
    }
  }

  return merged_clusters;
}

auto HSVClusterTrackerNode::GetClusters() const
    -> const std::vector<kmeans_cluster_t>* {
  return &clusters_;
}

}  // namespace gamepiece
