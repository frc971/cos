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

auto Covariance(const std::vector<cv::Point2f>& points) -> cv::Mat {
  cv::Point2f mean{0.0f, 0.0f};
  for (const cv::Point2f& point : points) {
    mean += point;
  }
  mean /= static_cast<float>(points.size());

  cv::Mat covariance = cv::Mat::zeros(2, 2, CV_32F);
  for (const cv::Point2f& point : points) {
    const cv::Point2f offset = point - mean;
    covariance.at<float>(0, 0) += offset.x * offset.x;
    covariance.at<float>(0, 1) += offset.x * offset.y;
    covariance.at<float>(1, 0) += offset.y * offset.x;
    covariance.at<float>(1, 1) += offset.y * offset.y;
  }
  return covariance / static_cast<float>(points.size());
}

}  // namespace

HSVClusterTrackerNode::HSVClusterTrackerNode(
    std::string_view input_path, std::string_view output_path,
    cv::Matx33d intrinsics, cv::Matx44d extrinsics,
    const cv::Vec<double, 5>& distortion_coeffs,
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
      max_ground_distance_m_(max_ground_distance_m) {
  if (!std::isfinite(max_ground_distance_m_) || max_ground_distance_m_ <= 0) {
    throw std::invalid_argument(
        "Maximum ground distance must be finite and positive");
  }
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

void HSVClusterTrackerNode::ProcessFrame(const cv::Mat& frame) {
  const std::vector<kmeans_cluster_t> previous_clusters = clusters_;
  clusters_.clear();
  thresholded_points_.clear();

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

  int cluster_count = kInitialClusterCount;
  if (!previous_clusters.empty()) {
    const std::vector<kmeans_cluster_t> assigned_clusters =
        AssignToExistingClusters(thresholded_points_, previous_clusters);
    cluster_count = static_cast<int>(previous_clusters.size()) +
                    CovarianceSpikeCount(previous_clusters, assigned_clusters);
  }
  cluster_count =
      std::min({active_cluster_count_,
                static_cast<int>(thresholded_points_.size()), cluster_count});
  const std::vector<kmeans_cluster_t> unmerged_clusters =
      KMeans(thresholded_points_, cluster_count, previous_clusters);
  clusters_ = MergeOverlappingClusters(unmerged_clusters);
  for (auto& cluster : clusters_) {
    cluster.camera_relative_translation.emplace(ClusterDistance(cluster));
  }
}

auto HSVClusterTrackerNode::AssignToExistingClusters(
    const std::vector<cv::Point2f>& data_points,
    const std::vector<kmeans_cluster_t>& existing_clusters) const
    -> std::vector<kmeans_cluster_t> {
  if (existing_clusters.empty()) {
    return {};
  }

  std::vector<cv::Point2f> scaled_centroids;
  std::vector<std::size_t> existing_cluster_indices;
  scaled_centroids.reserve(existing_clusters.size());
  existing_cluster_indices.reserve(existing_clusters.size());
  for (std::size_t cluster_index = 0; cluster_index < existing_clusters.size();
       ++cluster_index) {
    const kmeans_cluster_t& cluster = existing_clusters[cluster_index];
    const std::optional<frc::Translation2d> point_offset =
        NormalizedPointOffset(cluster.centroid);
    if (!point_offset.has_value()) {
      continue;
    }
    scaled_centroids.emplace_back(
        static_cast<float>(point_offset->X().value()),
        static_cast<float>(point_offset->Y().value()));
    existing_cluster_indices.push_back(cluster_index);
  }

  std::vector<kmeans_cluster_t> assigned_clusters(existing_clusters.size());
  for (std::size_t cluster_index = 0; cluster_index < existing_clusters.size();
       ++cluster_index) {
    assigned_clusters[cluster_index].centroid =
        existing_clusters[cluster_index].centroid;
  }
  if (scaled_centroids.empty()) {
    return assigned_clusters;
  }

  for (const cv::Point2f& point : data_points) {
    const std::optional<frc::Translation2d> point_offset =
        NormalizedPointOffset(point);
    if (!point_offset.has_value()) {
      continue;
    }
    const cv::Point2f scaled_point(
        static_cast<float>(point_offset->X().value()),
        static_cast<float>(point_offset->Y().value()));
    std::size_t nearest_center = 0;
    float nearest_distance = cv::norm(scaled_point - scaled_centroids.front());
    for (std::size_t center_index = 1; center_index < scaled_centroids.size();
         ++center_index) {
      const float distance =
          cv::norm(scaled_point - scaled_centroids[center_index]);
      if (distance < nearest_distance) {
        nearest_center = center_index;
        nearest_distance = distance;
      }
    }
    assigned_clusters[existing_cluster_indices[nearest_center]]
        .img_points.push_back(point);
  }

  for (kmeans_cluster_t& cluster : assigned_clusters) {
    if (cluster.img_points.size() > 1) {
      cluster.covar = Covariance(cluster.img_points);
    }
  }
  return assigned_clusters;
}

auto HSVClusterTrackerNode::CovarianceSpikeCount(
    const std::vector<kmeans_cluster_t>& previous_clusters,
    const std::vector<kmeans_cluster_t>& assigned_clusters) const -> int {
  int spike_count = 0;
  const std::size_t cluster_count =
      std::min(previous_clusters.size(), assigned_clusters.size());
  for (std::size_t cluster_index = 0; cluster_index < cluster_count;
       ++cluster_index) {
    const kmeans_cluster_t& previous = previous_clusters[cluster_index];
    const kmeans_cluster_t& assigned = assigned_clusters[cluster_index];
    if (previous.covar.rows != 2 || previous.covar.cols != 2 ||
        assigned.covar.rows != 2 || assigned.covar.cols != 2) {
      continue;
    }

    // manual calculation of trace for covariance
    const float previous_spread =
        previous.covar.at<float>(0, 0) + previous.covar.at<float>(1, 1);
    const float assigned_spread =
        assigned.covar.at<float>(0, 0) + assigned.covar.at<float>(1, 1);
    if (previous_spread > std::numeric_limits<float>::epsilon() &&
        assigned_spread > previous_spread * kCovarianceSpikeRatio) {
      ++spike_count;
    }
  }
  return spike_count;
}

auto HSVClusterTrackerNode::KMeans(
    const std::vector<cv::Point2f>& data_points, const int requested_k,
    const std::vector<kmeans_cluster_t>& initial_clusters) const
    -> std::vector<kmeans_cluster_t> {
  if (data_points.empty() || requested_k <= 0 ||
      std::cmp_greater(requested_k, data_points.size())) {
    throw std::invalid_argument("Invalid HSV KMeans configuration");
  }

  std::vector<cv::Point2f> scaled_points;
  std::vector<size_t> used_point_indices;
  scaled_points.reserve(data_points.size());
  for (size_t i = 0; i < data_points.size(); i++) {
    // inaccurate for most points because this assumes they're on the floor, may change later
    const std::optional<frc::Translation2d> point_offset =
        NormalizedPointOffset(data_points[i]);
    if (!point_offset.has_value()) {
      continue;  // to avoid yellow in the stands, which is above the field hoizon line
    }
    // Use meters in both dimensions. The floor projection accounts for the
    // larger ground distance represented by vertical motion in the image.
    scaled_points.emplace_back(static_cast<float>(point_offset->X().value()),
                               static_cast<float>(point_offset->Y().value()));
    used_point_indices.push_back(i);
  }
  if (scaled_points.empty()) {
    return {};
  }
  // Horizon rejection can leave fewer usable points than requested clusters.
  const int k = std::min(requested_k, static_cast<int>(scaled_points.size()));

  cv::Mat labels;
  cv::Mat centers;
  const cv::TermCriteria criteria(
      cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-4);

  std::vector<cv::Point2f> initial_centers;
  initial_centers.reserve(k);
  for (const kmeans_cluster_t& cluster : initial_clusters) {
    if (std::cmp_equal(initial_centers.size(), k)) {
      break;
    }
    const std::optional<frc::Translation2d> point_offset =
        NormalizedPointOffset(cluster.centroid);
    if (point_offset.has_value()) {
      initial_centers.emplace_back(
          static_cast<float>(point_offset->X().value()),
          static_cast<float>(point_offset->Y().value()));
    }
  }

  std::vector<bool> new_centroids(scaled_points.size(), false);
  while (std::cmp_less(initial_centers.size(), k)) {
    std::size_t farthest_point = scaled_points.size();
    float farthest_distance = -1.0f;
    for (std::size_t point_index = 0; point_index < scaled_points.size();
         ++point_index) {
      if (new_centroids[point_index]) {
        continue;
      }

      double distance_to_nearest_center = std::numeric_limits<double>::max();
      for (const cv::Point2f& center : initial_centers) {
        distance_to_nearest_center =
            std::min(distance_to_nearest_center,
                     cv::norm(scaled_points[point_index] - center));
      }
      if (initial_centers.empty() ||
          distance_to_nearest_center > farthest_distance) {
        farthest_point = point_index;
        farthest_distance = distance_to_nearest_center;
      }
    }

    if (farthest_point == scaled_points.size()) {
      break;
    }
    new_centroids[farthest_point] = true;
    initial_centers.push_back(scaled_points[farthest_point]);
  }

  labels = cv::Mat(static_cast<int>(scaled_points.size()), 1, CV_32S);
  std::vector<int> label_counts(k, 0);
  for (int point_index = 0; point_index < labels.rows; ++point_index) {
    int nearest_center = 0;
    float nearest_distance =
        cv::norm(scaled_points[point_index] - initial_centers.front());
    for (int center_index = 1; center_index < k; ++center_index) {
      const float distance =
          cv::norm(scaled_points[point_index] - initial_centers[center_index]);
      if (distance < nearest_distance) {
        nearest_center = center_index;
        nearest_distance = distance;
      }
    }
    labels.at<int>(point_index, 0) = nearest_center;
    ++label_counts[nearest_center];
  }

  for (int empty_cluster = 0; empty_cluster < k; ++empty_cluster) {
    if (label_counts[empty_cluster] != 0) {
      continue;
    }

    int point_to_reassign = -1;
    float closest_distance = std::numeric_limits<float>::max();
    for (int point_index = 0; point_index < labels.rows; ++point_index) {
      const int current_cluster = labels.at<int>(point_index, 0);
      if (label_counts[current_cluster] <= 1) {
        continue;
      }
      const float distance =
          cv::norm(scaled_points[point_index] - initial_centers[empty_cluster]);
      if (distance < closest_distance) {
        point_to_reassign = point_index;
        closest_distance = distance;
      }
    }

    if (point_to_reassign >= 0) {
      const int old_cluster = labels.at<int>(point_to_reassign, 0);
      labels.at<int>(point_to_reassign, 0) = empty_cluster;
      --label_counts[old_cluster];
      ++label_counts[empty_cluster];
    }
  }

  cv::kmeans(scaled_points, k, labels, criteria, 1,
             cv::KMEANS_USE_INITIAL_LABELS, centers);

  std::vector<cv::Point2f> centroid_sums(k, {0.0f, 0.0f});
  std::vector<int> centroid_counts(k, 0);

  for (int i = 0; i < labels.rows; ++i) {
    const int cluster = labels.at<int>(i, 0);
    centroid_sums[cluster] += data_points[used_point_indices[i]];
    ++centroid_counts[cluster];
  }

  std::vector<cv::Point2f> centroids(k);
  for (int i = 0; i < k; ++i) {
    centroids[i] = centroid_sums[i] / static_cast<float>(centroid_counts[i]);
  }

  std::vector<std::vector<cv::Point2f>> cluster_points(k);
  for (int i = 0; i < labels.rows; ++i) {
    const int cluster = labels.at<int>(i, 0);
    cluster_points.at(cluster).push_back(data_points.at(used_point_indices[i]));
  }

  std::vector<kmeans_cluster_t> clusters;
  clusters.reserve(k);
  for (int i = 0; i < k; ++i) {
    kmeans_cluster_t cluster;
    cluster.centroid = centroids[i];
    cluster.img_points = std::move(cluster_points.at(i));

    if (cluster.img_points.size() > 1) {
      cluster.covar = Covariance(cluster.img_points);
    } else {
      cluster.covar = cv::Mat::zeros(2, 2, CV_32F);
    }
    clusters.push_back(std::move(cluster));
  }
  return clusters;
}

auto HSVClusterTrackerNode::ClusterDistance(
    const kmeans_cluster_t& cluster) const -> frc::Translation2d {
  if (cluster.img_points.empty()) {
    return {};
  }

  const auto lowest_point = std::ranges::max_element(
      cluster.img_points,
      [](const cv::Point2f& first, const cv::Point2f& second) -> bool {
        return first.y < second.y;
      });
  const auto offset = NormalizedPointOffset(*lowest_point).value();
  return offset;
}

auto HSVClusterTrackerNode::ClustersOverlap(
    const kmeans_cluster_t& first, const kmeans_cluster_t& second) const
    -> bool {
  const auto make_ellipse = [](const kmeans_cluster_t& cluster) -> Ellipse {
    const cv::Mat& covariance = cluster.covar;
    if (covariance.rows != 2 || covariance.cols != 2) {
      throw std::invalid_argument("KMeans covariance must be 2 by 2");
    }

    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    cv::eigen(covariance, eigenvalues, eigenvectors);
    constexpr float kMinimumRadius = 1e-6f;
    const float major =
        std::sqrt(std::max(eigenvalues.at<float>(0, 0), 0.0f)) + kMinimumRadius;
    const float minor =
        std::sqrt(std::max(eigenvalues.at<float>(1, 0), 0.0f)) + kMinimumRadius;
    const cv::Vec2f major_axis(eigenvectors.at<float>(0, 0),
                               eigenvectors.at<float>(0, 1));
    return {.center = {cluster.centroid.x, cluster.centroid.y},
            .semi_axes = {major, minor},
            .rotation = std::atan2(major_axis[1], major_axis[0])};
  };

  return ellipse_overlap_area(make_ellipse(first), make_ellipse(second)) > 0.0f;
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
      merged.covar = cv::Mat::zeros(2, 2, CV_32F);
    }
  }

  return merged_clusters;
}

auto HSVClusterTrackerNode::GetClusters() const
    -> const std::vector<kmeans_cluster_t>* {
  return &clusters_;
}

}  // namespace gamepiece
