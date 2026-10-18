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
#include "localization/position.h"

namespace gamepiece {
namespace {

constexpr std::pair<int, int> hsv_color_range{18, 30};
constexpr int minimum_saturation{100};
constexpr float kFuelRadiusM = 0.075f;
constexpr float kCellM = 0.1f;
constexpr float kBandwidthM = 0.3f;
constexpr float kMergeBandwidthM = 0.4f;
constexpr float kModeOverlapD2 = 4.6051702f;  // 90% contour for a 2D Gaussian.

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
    const float max_ground_distance_m, std::string_view pose_path)
    : input_path_(input_path),
      output_path_(output_path),
      pose_path_(pose_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_(
          {control_loop::MessageDescriptor::Publication<hsv_clusters_t>(
              output_path_)}),
      camera_intrinsics_(intrinsics),
      distortion_coeffs_(distortion_coeffs),
      camera_extrinsics_cv_(extrinsics),
      max_ground_distance_m_(max_ground_distance_m) {
  if (image_size == 0) {
    throw std::invalid_argument("Image size must be positive");
  }
  if (!std::isfinite(max_ground_distance_m_) || max_ground_distance_m_ <= 0) {
    throw std::invalid_argument(
        "Maximum ground distance must be finite and positive");
  }
  if (!std::isfinite(extrinsics(2, 3)) || extrinsics(2, 3) <= kFuelRadiusM) {
    throw std::invalid_argument(
        "HSV tracker extrinsics must be the camera pose in robot coordinates "
        "above the Fuel center plane");
  }
  if (!pose_path_.empty()) dependencies_.emplace_back(
      pose_path_, typeid(localization::PositionEstimateMessage));
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
    std::optional<frc::Pose2d> pose = pose_path_.empty() ? std::nullopt : robot_pose_;
    if (!pose_path_.empty()) {
      const auto* estimate = context->GetMessage<localization::PositionEstimateMessage>(pose_path_);
      if (estimate) pose = estimate->pose.ToPose2d();
    }
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      ProcessFrame(cv::Mat{}, pose);
      context->SetMessage(output_path_, nullptr);
    } else {
      CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_BGRI);
      CHECK_GE(frame->stride, static_cast<size_t>(frame->width) * 3U);
      const cv::cuda::GpuMat image(frame->height, frame->width, CV_8UC3,
                                   frame->destination.channel[0],
                                   frame->stride);
      image.download(frame_);
      ProcessFrame(frame_, pose);
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
  if (robot_pose && (!std::isfinite(robot_pose->X().value()) ||
      !std::isfinite(robot_pose->Y().value()) ||
      !std::isfinite(robot_pose->Rotation().Radians().value())))
    throw std::invalid_argument("Robot pose must be finite");
  // Switching coordinate frames invalidates the old ID map.
  if (robot_pose.has_value() != robot_pose_.has_value()) {
    canonical_positions_.clear(); canonical_covariances_.clear();
  }
  robot_pose_ = robot_pose;
  clusters_.clear();
  thresholded_points_.clear();
  ground_points_.clear();
  ground_weights_.clear();
  refined_weights_.clear();
  if (frame.empty()) return;
  HSVThreshold(frame, hsv_color_range.first, hsv_color_range.second,
               minimum_saturation, thresholded_points_, hsv_image_, hsv_masked_);
  if (thresholded_points_.empty()) return;
  // Every silhouette tangent constrains a sphere center, without separating balls.
  // For unit ray u and inward perpendicular v, center = t*u + radius*v.
  // Its known height determines t. Occluded boundary arcs still provide evidence.
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(hsv_masked_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  thresholded_points_.clear();
  const cv::Matx33f rotation = camera_extrinsics_cv_.get_minor<3, 3>(0, 0);
  cv::Mat1f inside, outside;
  cv::distanceTransform(hsv_masked_, inside, cv::DIST_L2, 3);
  cv::distanceTransform(255 - hsv_masked_, outside, cv::DIST_L2, 3);
  // Verify tangent-derived sphere centers against the entire visible outline.
  // This also estimates the visible perimeter fraction for occlusion weighting.
  const auto visible_fraction = [&](const cv::Vec3f& center) {
    const float norm = cv::norm(center);
    const auto axis = center / norm;
    auto first = axis.cross(cv::Vec3f{0, 1, 0});
    first /= cv::norm(first);
    const auto second = axis.cross(first);
    const float sine = kFuelRadiusM / norm, cosine = std::sqrt(1 - sine * sine);
    int supported = 0, accessible = 0, hidden = 0;
    constexpr int kSamples = 32;
    for (int j = 0; j < kSamples; ++j) {
      const float angle = 2 * std::numbers::pi_v<float> * j / kSamples;
      const auto direction = cosine * axis + sine * (std::cos(angle) * first + std::sin(angle) * second);
      const float x = direction[0] / direction[2], y = direction[1] / direction[2];
      const float ux = x * camera_intrinsics_(0, 0) + camera_intrinsics_(0, 2);
      const float uy = y * camera_intrinsics_(1, 1) + camera_intrinsics_(1, 2);
      if (ux < 2 || ux >= frame.cols - 2 || uy < 2 || uy >= frame.rows - 2) continue;
      const float radius2 = x * x + y * y;
      const float radial = 1 + radius2 * (distortion_coeffs_[0] + radius2 *
          (distortion_coeffs_[1] + radius2 * distortion_coeffs_[4]));
      const float dx = x * radial + 2 * distortion_coeffs_[2] * x * y + distortion_coeffs_[3] * (radius2 + 2 * x * x);
      const float dy = y * radial + distortion_coeffs_[2] * (radius2 + 2 * y * y) + 2 * distortion_coeffs_[3] * x * y;
      const int px = cvRound(dx * camera_intrinsics_(0, 0) + camera_intrinsics_(0, 2));
      const int py = cvRound(dy * camera_intrinsics_(1, 1) + camera_intrinsics_(1, 2));
      if (px < 0 || px >= frame.cols || py < 0 || py >= frame.rows) continue;
      ++accessible;
      if (std::max(inside(py, px), outside(py, px)) <= 2.0f) ++supported;
      if (inside(py, px) > 2.0f) ++hidden;
    }
    // Clipped arcs cannot contradict a candidate; verify the accessible outline.
    // Only arcs inside yellow or outside the image can be occluded. An arc
    // outside yellow is a fitting error and must not increase estimated mass.
    return supported >= 4 && supported >= 0.25f * accessible ?
        static_cast<float>(accessible - hidden) / kSamples : 0.0f;
  };
  for (const auto& contour : contours) {
    if (contour.size() < 12 || cv::contourArea(contour) < 8) continue;
    std::vector<cv::Point2f> pixels(contour.begin(), contour.end()), normalized;
    cv::undistortPoints(pixels, normalized, camera_intrinsics_, distortion_coeffs_);
    std::vector<cv::Vec3f> rays;
    for (const auto& p : normalized) {
      const cv::Vec3f ray{p.x, p.y, 1};
      rays.push_back(ray / cv::norm(ray));
    }
    const auto clipped = [&](std::size_t i) {
      const auto& p = pixels[i];
      const float ux = normalized[i].x * camera_intrinsics_(0, 0) + camera_intrinsics_(0, 2);
      const float uy = normalized[i].y * camera_intrinsics_(1, 1) + camera_intrinsics_(1, 2);
      return p.x < 1 || p.x >= frame.cols - 1 || p.y < 1 || p.y >= frame.rows - 1 ||
          ux < 1 || ux >= frame.cols - 1 || uy < 1 || uy >= frame.rows - 1;
    };
    for (std::size_t i = 0; i < rays.size(); ++i) {
      // Image clipping is not a sphere tangent. The simulator's distortion
      // remap can also clip the source. Check the whole tangent stencil.
      const auto before = (i + rays.size() - 4) % rays.size(), after = (i + 4) % rays.size();
      if (clipped(i) || clipped(before) || clipped(after)) continue;
      const auto tangent = rays[after] - rays[before];
      auto normal = tangent.cross(rays[i]);
      const float length = cv::norm(normal);
      if (length < 1e-6f) continue;
      const auto camera_normal = normal / length;
      normal = rotation * camera_normal;
      const auto ray = rotation * rays[i];
      if (ray[1] <= 1e-5f) continue;
      const float distance = (-kFuelRadiusM - camera_extrinsics_cv_(1, 3) -
                              kFuelRadiusM * normal[1]) / ray[1];
      if (distance <= 0 || distance > max_ground_distance_m_) continue;
      const float fraction = visible_fraction(distance * rays[i] + kFuelRadiusM * camera_normal);
      const auto center = distance * ray + kFuelRadiusM * normal;
      const auto offset = frc::Translation2d{units::meter_t{center[2]},
                                             units::meter_t{-center[0]}} + extrinsics_offset_;
      const auto position = robot_pose_ ? robot_pose_->Translation() +
          offset.RotateBy(robot_pose_->Rotation()) : offset;
      const cv::Point2f ground{static_cast<float>(position.X().value()),
                               static_cast<float>(position.Y().value())};
      if (robot_pose_ && (ground.x < 4.6f || ground.x > 11.94f ||
                          ground.y < 0.0f || ground.y > 8.07f)) continue;
      // Inverse angular perimeter balances near/far silhouettes; hidden arcs
      // contribute no mass, so fuel_count estimates visible support, not inventory.
      const float weight = distance / (2 * std::numbers::pi_v<float> *
          kFuelRadiusM * std::sqrt(camera_intrinsics_(0, 0) * camera_intrinsics_(1, 1)));
      thresholded_points_.push_back(normalized[i]);
      ground_points_.push_back(ground);
      ground_weights_.push_back(weight);
      refined_weights_.push_back(fraction > 0 ? weight / fraction : 0.0f);
    }
  }
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
    // Verified outlines carry geometry; retain weak votes for partial entrants.
    density(cell / width, cell % width) +=
        std::max(refined_weights_[i], 0.1f * ground_weights_[i]);
  }
  cv::Mat1f coarse;
  cv::GaussianBlur(density, coarse, {}, kMergeBandwidthM / kCellM);
  cv::GaussianBlur(density, density, {}, kBandwidthM / kCellM);
  const int count = width * height;
  std::vector<int> order(count), basin(count, -1);
  std::iota(order.begin(), order.end(), 0);
  const auto value = [&](int cell) { return density(cell / width, cell % width); };
  const auto merge_value = [&](int cell) { return coarse(cell / width, cell % width); };
  std::ranges::sort(order, [&](int a, int b) { return value(a) > value(b); });
  std::vector<int> prior(count, -1), peak_cell(count);
  std::iota(peak_cell.begin(), peak_cell.end(), 0);
  std::vector<float> peak(count);
  for (int cell : order) peak[cell] = value(cell);
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
    if (best == cell) {
      const cv::Point2f center = origin + cv::Point2f{(x + 0.5f) * kCellM, (y + 0.5f) * kCellM};
      float closest = 4.0f;
      for (std::size_t id = 0; id < canonical_positions_.size(); ++id) {
        const cv::Vec2f delta{center.x - static_cast<float>(canonical_positions_[id].X().value()),
                              center.y - static_cast<float>(canonical_positions_[id].Y().value())};
        const auto inverse = (canonical_covariances_[id] + cv::Matx22f::eye() * 0.16f).inv();
        const float distance = delta.dot(inverse * delta);
        if (distance < closest) { closest = distance; prior[cell] = id; }
      }
    }
  }
  struct Moments { float mass = 0; cv::Vec2f sum{}; cv::Matx22f square{}; };
  std::vector<Moments> moments(count);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (basin[cells[i]] < 0) continue;
    auto& m = moments[basin[cells[i]]];
    const float w = std::max(refined_weights_[i], 0.1f * ground_weights_[i]);
    const cv::Vec2f p{ground_points_[i].x, ground_points_[i].y};
    m.mass += w; m.sum += w * p; m.square += w * (p * p.t());
  }
  for (int cell : order) {
    if (basin[cell] < 0) continue;
    const int x = cell % width, y = cell / width;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
      const int nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const int neighbor = ny * width + nx;
      if (basin[neighbor] < 0) continue;
      const auto first = modes.Find(basin[cell]), second = modes.Find(basin[neighbor]);
      if (first == second) continue;
      const int pa = peak_cell[first], pb = peak_cell[second];
      const float separation = kCellM * std::hypot(pa % width - pb % width, pa / width - pb / width);
      // A new density mode beyond an established pile's 2-sigma footprint
      // must not inflate that pile merely because its visible mass is small.
      const bool different_piles = prior[first] != prior[second] &&
          (prior[first] >= 0 || prior[second] >= 0) && separation >= 2 * kBandwidthM;
      const auto a = moments[first], b = moments[second];
      bool overlap = a.mass == 0 || b.mass == 0 ||
          (prior[first] >= 0 && prior[first] == prior[second]);
      if (!overlap) {
        const auto ma = a.sum / a.mass, mb = b.sum / b.mass, delta = ma - mb;
        const auto ca = a.square / a.mass - ma * ma.t() + cv::Matx22f::eye() * (kFuelRadiusM * kFuelRadiusM);
        const auto cb = b.square / b.mass - mb * mb.t() + cv::Matx22f::eye() * (kFuelRadiusM * kFuelRadiusM);
        // Compare footprints at the pile resolution, rather than pixel noise.
        const auto covariance = ca + cb + cv::Matx22f::eye() *
            (2 * (kBandwidthM * kBandwidthM - kFuelRadiusM * kFuelRadiusM));
        overlap = delta.dot(covariance.inv() * delta) <= kModeOverlapD2;
        if (!overlap) {
          // Distant sampling lobes of one elongated Gaussian can still belong
          // together. Compare their weighted classification likelihoods.
          const float n = a.mass + b.mass;
          const auto mean = (a.sum + b.sum) / n;
          const auto merged = (a.square + b.square) / n - mean * mean.t() + cv::Matx22f::eye() * (kFuelRadiusM * kFuelRadiusM);
          const double gain = 0.5 * (n * std::log(cv::determinant(merged)) - a.mass * std::log(cv::determinant(ca)) - b.mass * std::log(cv::determinant(cb))) +
              a.mass * std::log(a.mass / n) + b.mass * std::log(b.mass / n);
          overlap = gain <= 0;
        }
      }
      // Resolve weak entrants at 30 cm, but require a valley at 40 cm to split
      // untracked piles: sampling bumps along a broad pile should merge.
      if (!different_piles && overlap && std::min(merge_value(cell), merge_value(neighbor)) >=
          0.75f * std::min(merge_value(pa), merge_value(pb))) {
        const bool keep_first_peak = peak[first] >= peak[second];
        const float merged_peak = std::max(peak[first], peak[second]);
        const int merged_cell = keep_first_peak ? pa : pb;
        const int merged_prior = prior[first] >= 0 ? prior[first] : prior[second];
        modes.Union(first, second);
        const auto root = modes.Find(first);
        peak[root] = merged_peak; peak_cell[root] = merged_cell; prior[root] = merged_prior;
        moments[root] = {a.mass + b.mass, a.sum + b.sum, a.square + b.square};
      }
    }
  }
  std::vector<kmeans_cluster_t> result;
  std::vector<std::vector<float>> raw_weights;
  std::vector<int> indices(count, -1);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (basin[cells[i]] < 0) continue;
    const int root = modes.Find(basin[cells[i]]);
    if (indices[root] < 0) { indices[root] = result.size(); result.emplace_back(); raw_weights.emplace_back(); }
    auto& cluster = result[indices[root]];
    cluster.img_points.push_back(thresholded_points_[i]);
    cluster.ground_points.push_back(ground_points_[i]);
    cluster.ground_weights.push_back(refined_weights_[i]);
    raw_weights[indices[root]].push_back(ground_weights_[i]);
    cluster.fuel_count += ground_weights_[i];
  }
  for (std::size_t index = 0; index < result.size(); ++index) {
    auto& c = result[index];
    const float refined_mass = std::accumulate(c.ground_weights.begin(), c.ground_weights.end(), 0.0f);
    // Occluded piles may have enough verified support but few exposed arcs.
    if (std::max(c.fuel_count, refined_mass) < 1.5f) {
      c.img_points.clear(); continue;
    }
    if (refined_mass < 0.25f * c.fuel_count) c.ground_weights = std::move(raw_weights[index]);
    else c.fuel_count = refined_mass;
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
  std::erase_if(result, [](const auto& c) { return c.img_points.empty(); });
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
  std::vector<cv::Matx22f> covariances;
  const float angle = robot_pose_ ? robot_pose_->Rotation().Radians().value() : 0.0f;
  const cv::Matx22f rotation{std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle)};
  for (const auto& cluster : clusters_) covariances.push_back(rotation * cluster.ground_covar * rotation.t());
  std::ranges::sort(matches);
  std::vector<bool> used(canonical_positions_.size(), false), assigned(clusters_.size(), false);
  for (const auto& [distance, i, id] : matches) {
    if (used[id] || assigned[i]) continue;
    clusters_[i].id = id;
    used[id] = assigned[i] = true;
    canonical_positions_[id] = positions[i];
    canonical_covariances_[id] = covariances[i];
  }
  for (std::size_t i = 0; i < clusters_.size(); ++i) if (!assigned[i]) {
    clusters_[i].id = canonical_positions_.size();
    canonical_positions_.push_back(positions[i]);
    canonical_covariances_.push_back(covariances[i]);
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
