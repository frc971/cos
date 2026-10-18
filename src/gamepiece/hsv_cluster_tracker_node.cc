#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <numbers>
#include <tuple>
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
constexpr int minimum_saturation{100};
constexpr float kFuelRadiusM = 0.075f;
constexpr float kCellM = 0.1f;
constexpr float kBandwidthM = 0.4f;

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
    const cv::Point2f& point) const -> frc::Translation2d {
  return UndistortedPinholePointOffset(point, -kFuelRadiusM, camera_extrinsics_cv_);
}

auto HSVClusterTrackerNode::RobotRelativeOffset(const cv::Point2f& point) const
    -> frc::Translation2d {
  return NormalizedPointOffset(point) + extrinsics_offset_;
}

void HSVClusterTrackerNode::ProcessFrame(
    const cv::Mat& frame, std::optional<frc::Pose2d> robot_pose) {
  // Switching coordinate frames invalidates the old ID map.
  if (robot_pose.has_value() != robot_pose_.has_value()) canonical_positions_.clear();
  robot_pose_ = robot_pose;
  clusters_.clear();
  thresholded_points_.clear();
  ground_points_.clear();
  ground_weights_.clear();
  if (frame.empty()) return;
  HSVThreshold(frame, hsv_color_range.first, hsv_color_range.second,
               minimum_saturation, thresholded_points_, hsv_image_, hsv_masked_);
  if (thresholded_points_.empty()) return;
  cv::undistortPoints(thresholded_points_, thresholded_points_,
                      camera_intrinsics_, distortion_coeffs_);
  CullAboveHorizon(thresholded_points_, camera_extrinsics_cv_);
  std::vector<cv::Point2f> retained;
  for (const auto& point : thresholded_points_) {
    const auto offset = RobotRelativeOffset(point);
    if (!std::isfinite(offset.Norm().value()) ||
        offset.Distance(extrinsics_offset_).value() > max_ground_distance_m_) continue;
    const auto position = robot_pose_ ? robot_pose_->Translation() +
        offset.RotateBy(robot_pose_->Rotation()) : offset;
    const cv::Point2f ground{static_cast<float>(position.X().value()),
                             static_cast<float>(position.Y().value())};
    // The neutral zone lies between the alliance-side trench faces.
    if (robot_pose_ && (ground.x < 4.6f || ground.x > 11.94f ||
                        ground.y < 0.0f || ground.y > 8.07f)) continue;
    const float ray_y = camera_extrinsics_cv_(1, 0) * point.x +
        camera_extrinsics_cv_(1, 1) * point.y + camera_extrinsics_cv_(1, 2);
    const float depth = (-kFuelRadiusM - camera_extrinsics_cv_(1, 3)) / ray_y;
    // Undo perspective area: a ball covers approximately pi*(f*r/depth)^2 pixels.
    const float weight = depth * depth / (std::numbers::pi_v<float> *
        kFuelRadiusM * kFuelRadiusM * camera_intrinsics_(0, 0) * camera_intrinsics_(1, 1));
    retained.push_back(point);
    ground_points_.push_back(ground);
    ground_weights_.push_back(weight);
  }
  thresholded_points_ = std::move(retained);
  if (ground_points_.empty()) return;
  clusters_ = ClusterGroundPoints();
  AssignPersistentIds();
}

auto HSVClusterTrackerNode::ClusterGroundPoints() -> std::vector<kmeans_cluster_t> {
  // A fixed physical bandwidth groups piles rather than individual ball silhouettes.
  cv::Point2f origin = ground_points_.front(), maximum = origin;
  for (const auto& p : ground_points_) {
    origin.x = std::min(origin.x, p.x); origin.y = std::min(origin.y, p.y);
    maximum.x = std::max(maximum.x, p.x); maximum.y = std::max(maximum.y, p.y);
  }
  origin -= cv::Point2f{1.5f, 1.5f};
  const int width = static_cast<int>((maximum.x - origin.x + 1.5f) / kCellM) + 1;
  const int height = static_cast<int>((maximum.y - origin.y + 1.5f) / kCellM) + 1;
  cv::Mat1f density(height, width, 0.0f);
  std::vector<int> cells;
  for (std::size_t i = 0; i < ground_points_.size(); ++i) {
    const auto p = (ground_points_[i] - origin) / kCellM;
    const int cell = static_cast<int>(p.y) * width + static_cast<int>(p.x);
    cells.push_back(cell);
    density(cell / width, cell % width) += ground_weights_[i];
  }
  cv::GaussianBlur(density, density, {}, kBandwidthM / kCellM);
  const int count = width * height;
  std::vector<int> order(count), basin(count, -1);
  std::iota(order.begin(), order.end(), 0);
  const auto value = [&](int cell) { return density(cell / width, cell % width); };
  std::ranges::sort(order, [&](int a, int b) { return value(a) > value(b); });
  utils::DisjointSetUnion modes;
  modes.FillSets(count);
  // Descending watershed; join peaks only across a substantial density saddle.
  const float floor = value(order.front()) * 0.015f;
  for (int cell : order) {
    if (value(cell) < floor) break;
    const int x = cell % width, y = cell / width;
    int best = cell;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
      const int nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const int neighbor = ny * width + nx;
      if (value(neighbor) > value(best)) best = neighbor;
    }
    basin[cell] = best == cell ? cell : basin[best];
    if (basin[cell] < 0) continue;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
      const int nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const int neighbor = ny * width + nx;
      if (basin[neighbor] >= 0 && value(cell) >=
          0.55f * std::min(value(basin[cell]), value(basin[neighbor])))
        modes.Union(basin[cell], basin[neighbor]);
    }
  }
  std::vector<kmeans_cluster_t> result;
  std::vector<int> indices(count, -1);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (basin[cells[i]] < 0) continue;
    const int root = modes.Find(basin[cells[i]]);
    if (indices[root] < 0) { indices[root] = result.size(); result.emplace_back(); }
    auto& cluster = result[indices[root]];
    cluster.img_points.push_back(thresholded_points_[i]);
    cluster.ground_points.push_back(ground_points_[i]);
    cluster.ground_weights.push_back(ground_weights_[i]);
    cluster.fuel_count += ground_weights_[i];
  }
  std::erase_if(result, [this](const auto& c) {
    return c.fuel_count < 2.5f || c.img_points.size() < kmin_pixel_ratio * image_size_;
  });
  for (auto& c : result) {
    cv::Point2f center{};
    for (std::size_t i = 0; i < c.ground_points.size(); ++i) {
      center += c.ground_points[i] * c.ground_weights[i];
      c.centroid += c.img_points[i];
    }
    center /= c.fuel_count;
    c.centroid /= static_cast<float>(c.img_points.size());
    c.covar = Covariance(c.img_points);
    for (std::size_t i = 0; i < c.ground_points.size(); ++i) {
      const auto delta = c.ground_points[i] - center;
      c.ground_covar += cv::Matx22f{delta.x * delta.x, delta.x * delta.y,
                                   delta.x * delta.y, delta.y * delta.y} *
                       (c.ground_weights[i] / c.fuel_count);
    }
    auto position = frc::Translation2d{units::meter_t{center.x}, units::meter_t{center.y}};
    if (robot_pose_) position = (position - robot_pose_->Translation()).RotateBy(-robot_pose_->Rotation());
    c.robot_relative_translation = position;
    // Store public samples and covariance in robot coordinates.
    if (robot_pose_) {
      const float a = -robot_pose_->Rotation().Radians().value();
      const cv::Matx22f rotation{std::cos(a), -std::sin(a), std::sin(a), std::cos(a)};
      c.ground_covar = rotation * c.ground_covar * rotation.t();
      for (auto& p : c.ground_points) {
        const auto q = rotation * cv::Vec2f{p.x - static_cast<float>(robot_pose_->X().value()),
                                           p.y - static_cast<float>(robot_pose_->Y().value())};
        p = {q[0], q[1]};
      }
    }
  }
  return result;
}

void HSVClusterTrackerNode::AssignPersistentIds() {
  std::vector<frc::Translation2d> positions;
  std::vector<std::tuple<double, std::size_t, std::size_t>> matches;
  for (std::size_t i = 0; i < clusters_.size(); ++i) {
    const auto offset = *clusters_[i].robot_relative_translation;
    positions.push_back(robot_pose_ ? robot_pose_->Translation() +
        offset.RotateBy(robot_pose_->Rotation()) : offset);
    for (std::size_t id = 0; id < canonical_positions_.size(); ++id) {
      const double distance = positions.back().Distance(canonical_positions_[id]).value();
      if (distance <= kIdMatchToleranceM) matches.emplace_back(distance, i, id);
    }
  }
  std::ranges::sort(matches);
  std::vector<bool> used(canonical_positions_.size(), false), assigned(clusters_.size(), false);
  for (const auto& [distance, i, id] : matches) {
    if (used[id] || assigned[i]) continue;
    clusters_[i].id = id;
    used[id] = assigned[i] = true;
    canonical_positions_[id] = positions[i];
  }
  for (std::size_t i = 0; i < clusters_.size(); ++i) if (!assigned[i]) {
    clusters_[i].id = canonical_positions_.size();
    canonical_positions_.push_back(positions[i]);
  }
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

auto HSVClusterTrackerNode::GetClusters() const
    -> const std::vector<kmeans_cluster_t>* {
  return &clusters_;
}

}  // namespace gamepiece
