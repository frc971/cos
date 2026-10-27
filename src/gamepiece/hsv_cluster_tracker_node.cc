#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <frc/geometry/struct/Pose2dStruct.h>

#include "absl/log/check.h"
#include "camera/nvjpeg_decode_node.h"
#include "gamepiece/image_utils.h"
#include "localization/position.h"
#include "utils/cv_geometry.h"

namespace gamepiece {
namespace {

constexpr float kCellM = 0.1f;
constexpr float kBandwidthM = 0.3f;
constexpr float kMergeBandwidthM = 0.4f;
constexpr float kModeOverlapD2 = 4.6051702f;  // 90% contour for a 2D Gaussian.
}  // namespace

auto HSVClusterTrackerNode::Covariance(const std::vector<cv::Point2f>& points) -> cv::Matx22f {
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

HSVClusterTrackerNode::HSVClusterTrackerNode(
    std::string_view input_path, std::string_view output_path,
    cv::Matx33d intrinsics, cv::Matx44d extrinsics, std::size_t image_size,
    std::string_view pose_path, DensityHistory density_history,
    bool residual_density, std::string_view pose_topic,
    nt::NetworkTableInstance instance)
    : strategy_(std::make_unique<AnalyticalClusterStrategy>()),
      temporal_density_(density_history != DensityHistory::kNone),
      predictive_density_(density_history == DensityHistory::kPredict),
      residual_density_(residual_density),
      input_path_(input_path),
      output_path_(output_path),
      pose_path_(pose_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      // Intermediate clusters contain image samples, not generic log fields.
      publications_({{output_path_,
                      typeid(control_loop::ValueMessage<hsv_clusters_t>)}}),
      camera_intrinsics_(intrinsics),
      camera_extrinsics_cv_(extrinsics) {
  if (image_size == 0) {
    throw std::invalid_argument("Image size must be positive");
  }
  if (!std::isfinite(extrinsics(2, 3)) || extrinsics(2, 3) <= kFuelRadiusM) {
    throw std::invalid_argument(
        "HSV tracker extrinsics must be the camera pose in robot coordinates "
        "above the Fuel center plane");
  }
  if (!pose_path_.empty())
    dependencies_.emplace_back(pose_path_,
                               typeid(localization::PositionEstimateMessage));
  else if (!pose_topic.empty())
    pose_subscriber_ = instance.GetStructTopic<frc::Pose2d>(pose_topic).Subscribe(
        {}, {.periodic = 0.001});
  // Capture the ground translation before changing from WPILib coordinates.
  extrinsics_offset_ = frc::Translation2d{units::meter_t{extrinsics(0, 3)},
                                          units::meter_t{extrinsics(1, 3)}};
  utils::ChangeBasis(camera_extrinsics_cv_, utils::Basis::kWpiToCv);
  camera_position_ =
      cv::Vec3f{camera_extrinsics_cv_(0, 3), camera_extrinsics_cv_(1, 3),
                camera_extrinsics_cv_(2, 3)};
}

void HSVClusterTrackerNode::SetStrategy(std::unique_ptr<ClusterStrategy> strategy) {
  if (!strategy) throw std::invalid_argument("A cluster strategy is required");
  std::scoped_lock lock(processing_mutex_);
  strategy_ = std::move(strategy);
  canonical_positions_.clear();
  canonical_covariances_.clear();
  ball_tracks_.clear();
  density_states_.clear();
  clusters_.clear();
  hsv_masked_.release();
}

void HSVClusterTrackerNode::EnablePoseOutput(std::string_view output_path) {
  if (output_path.empty() || output_path == input_path_ ||
      output_path == output_path_ || output_path == mask_output_path_ || output_path == pose_path_ ||
      !pose_output_path_.empty()) {
    throw std::invalid_argument("Tracker pose output requires a distinct path");
  }
  pose_output_path_ = output_path;
  publications_.push_back(control_loop::MessageDescriptor::Publication<
      localization::PositionEstimateMessage>(pose_output_path_));
}

void HSVClusterTrackerNode::EnableMaskOutput(std::string_view output_path) {
  if (output_path.empty() || output_path == input_path_ || output_path == output_path_ ||
      output_path == pose_path_ || output_path == pose_output_path_ || !mask_output_path_.empty())
    throw std::invalid_argument("Tracker mask output requires a distinct path");
  mask_output_path_ = output_path;
  publications_.push_back(control_loop::MessageDescriptor::Publication<cv::Mat>(mask_output_path_));
}

auto HSVClusterTrackerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    std::unique_lock processing_lock(processing_mutex_);
    if (!context->Exists(input_path_) ||
        (!pose_path_.empty() && !context->Exists(pose_path_))) {
      return;
    }
    if (context->Exists(output_path_)) {
      return;
    }
    std::optional<frc::Pose2d> pose = last_robot_pose_;
    if (!pose_path_.empty()) {
      const auto* estimate =
          context->GetMessage<localization::PositionEstimateMessage>(
              pose_path_);
      if (estimate)
        pose = estimate->pose.ToPose2d();
    } else if (pose_subscriber_) {
      const auto latest = pose_subscriber_.GetAtomic();
      const auto& candidate = latest.value;
      if (latest.time > 0 && std::isfinite(candidate.X().value()) &&
          std::isfinite(candidate.Y().value()) &&
          std::isfinite(candidate.Rotation().Radians().value()))
        pose = candidate;
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
      ProcessFrame(image, pose);
      context->SetMessage(
          output_path_,
          std::make_shared<control_loop::ValueMessage<hsv_clusters_t>>(
              clusters_));
    }
    if (!pose_output_path_.empty()) {
      std::shared_ptr<localization::PositionEstimateMessage> estimate;
      if (pose) {
        estimate = std::make_shared<localization::PositionEstimateMessage>();
        estimate->pose = frc::Pose3d{*pose};
        if (frame) estimate->timestamp = frame->timestamp;
      }
      context->SetMessage(pose_output_path_, std::move(estimate));
    }
    if (!mask_output_path_.empty()) {
      context->SetMessage(mask_output_path_,
          std::make_shared<control_loop::ValueMessage<cv::Mat>>(
              frame && frame->destination.channel[0] ? hsv_masked_.clone() : cv::Mat{}));
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
  return UndistortedPinholePointOffset(point, -kFuelRadiusM,
                                       camera_extrinsics_cv_);
}

auto HSVClusterTrackerNode::RobotRelativeOffset(const cv::Point2f& point) const
    -> frc::Translation2d {
  return NormalizedPointOffset(point) + extrinsics_offset_;
}

void HSVClusterTrackerNode::ProcessFrame(
    cv::InputArray frame, std::optional<frc::Pose2d> robot_pose) {
  // Switching coordinate frames invalidates the old ID map.
  if (robot_pose.has_value() != last_robot_pose_.has_value()) {
    canonical_positions_.clear();
    canonical_covariances_.clear();
    ball_tracks_.clear();
    density_states_.clear();
  }
  for (auto& track : ball_tracks_) {
    // TODO move
    if (frame.empty())
      continue;
    const frc::Translation2d stored{units::meter_t{track.position.x},
                                    units::meter_t{track.position.y}};
    const auto robot_relative_position =
        robot_pose ? (stored - robot_pose->Translation())
                         .RotateBy(-robot_pose->Rotation())
                   : stored;
    const auto center =
        camera_extrinsics_cv_.get_minor<3, 3>(0, 0).t() *
        (cv::Vec3f{-static_cast<float>(robot_relative_position.Y().value()),
                   -kFuelRadiusM,
                   static_cast<float>(robot_relative_position.X().value())} -
         cv::Vec3f{camera_extrinsics_cv_(0, 3), camera_extrinsics_cv_(1, 3),
                   camera_extrinsics_cv_(2, 3)});
    if (center[2] <= kFuelRadiusM)
      continue;
    const float px = center[0] / center[2] * camera_intrinsics_(0, 0) +
                     camera_intrinsics_(0, 2);
    const float py = center[1] / center[2] * camera_intrinsics_(1, 1) +
                     camera_intrinsics_(1, 2);
    const float margin =
        2 * camera_intrinsics_(0, 0) * kFuelRadiusM / center[2];
    if (px >= -margin && px < frame.size().width + margin && py >= -margin &&
        py < frame.size().height + margin)
      ++track.missed;
  }
  std::erase_if(ball_tracks_, [](const auto& t) { return t.missed > 8; });
  for (auto& state : density_states_)
    ++state.missed;
  last_robot_pose_ = robot_pose;
  clusters_.clear();
  if (frame.empty()) {
    hsv_masked_.release();
    return;
  }
  strategy_->Process(*this, frame, robot_pose);
}

void HSVClusterTrackerNode::UpdateDensity(kmeans_cluster_t& c, cv::Matx22f covariance,
                                       const std::optional<frc::Pose2d>& robot_pose) {
  c.instantaneous_density_area_m2 =
      2 * std::numbers::pi_v<float> *
      std::sqrt(cv::determinant(covariance +
                                cv::Matx22f::eye() *
                                    (kFuelRadiusM * kFuelRadiusM / 4)));
  c.estimated_ball_count = c.instantaneous_ball_count;
  if (temporal_density_) {
    const float angle =
        robot_pose ? robot_pose->Rotation().Radians().value() : 0;
    const cv::Matx22f field_rotation{std::cos(angle), -std::sin(angle),
                                     std::sin(angle), std::cos(angle)};
    covariance = field_rotation * covariance * field_rotation.t();
    if (density_states_.size() <= c.id)
      density_states_.resize(c.id + 1);
    auto& state = density_states_[c.id];
    if (state.updates == 0 || state.missed > 8) {
      state.count = c.instantaneous_ball_count;
      state.covariance = covariance;
      state.updates = state.changed = 0;
      state.innovation = 0;
    } else {
      const float innovation = c.instantaneous_ball_count - state.count;
      const float threshold = state.changed >= 3 ? .08f : .25f;
      state.changed =
          std::abs(innovation) > threshold * std::max(2.0f, state.count)
              ? (innovation * state.innovation > 0 ? state.changed + 1 : 1)
              : 0;
      const float gain = state.changed >= 3 ? .65f : .20f;
      state.count += gain * innovation;
      state.covariance += gain * (covariance - state.covariance);
      state.innovation = innovation;
    }
    ++state.updates;
    state.missed = 0;
    c.estimated_ball_count =
        std::max(static_cast<float>(c.verified_ball_count), state.count);
    covariance = state.covariance;
  }
  c.density_area_m2 =
      2 * std::numbers::pi_v<float> *
      std::sqrt(cv::determinant(covariance +
                                cv::Matx22f::eye() *
                                    (kFuelRadiusM * kFuelRadiusM / 4)));
  c.density = c.estimated_ball_count / c.density_area_m2;
}

auto HSVClusterTrackerNode::ClusterGroundPoints(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point2f>& ground_points,
    const std::vector<float>& ground_weights,
    const std::vector<float>& refined_weights,
    const std::optional<frc::Pose2d>& robot_pose) const -> hsv_clusters_t {
  // A fixed physical bandwidth groups piles rather than individual ball silhouettes.
  cv::Point2f origin = ground_points.front(), maximum = origin;
  for (const auto& p : ground_points) {
    origin.x = std::min(origin.x, p.x);
    origin.y = std::min(origin.y, p.y);
    maximum.x = std::max(maximum.x, p.x);
    maximum.y = std::max(maximum.y, p.y);
  }
  origin -= cv::Point2f{1.5f, 1.5f};
  const int width =
      static_cast<int>((maximum.x - origin.x + 1.5f) / kCellM) + 1;
  const int height =
      static_cast<int>((maximum.y - origin.y + 1.5f) / kCellM) + 1;
  cv::Mat1f density(height, width, 0.0f);
  std::vector<int> cells;
  for (std::size_t i = 0; i < ground_points.size(); ++i) {
    const auto p = (ground_points[i] - origin) / kCellM;
    const int cell = static_cast<int>(p.y) * width + static_cast<int>(p.x);
    cells.push_back(cell);
    // Verified outlines carry geometry; retain weak votes for partial entrants.
    density(cell / width, cell % width) +=
        std::max(refined_weights[i], 0.1f * ground_weights[i]);
  }
  cv::Mat1f coarse;
  cv::GaussianBlur(density, coarse, {}, kMergeBandwidthM / kCellM);
  cv::GaussianBlur(density, density, {}, kBandwidthM / kCellM);
  const int count = width * height;
  std::vector<int> order(count), basin(count, -1);
  std::iota(order.begin(), order.end(), 0);
  const auto value = [&](int cell) {
    return density(cell / width, cell % width);
  };
  const auto merge_value = [&](int cell) {
    return coarse(cell / width, cell % width);
  };
  std::ranges::sort(order, [&](int a, int b) { return value(a) > value(b); });
  std::vector<int> prior(count, -1), peak_cell(count);
  std::iota(peak_cell.begin(), peak_cell.end(), 0);
  std::vector<float> peak(count);
  for (int cell : order)
    peak[cell] = value(cell);
  utils::DisjointSetUnion modes;
  modes.FillSets(count);
  // Descending watershed; join peaks only across a substantial density saddle.
  const float floor = value(order.front()) * 0.015f;
  for (int cell : order) {
    if (value(cell) < floor)
      break;
    const int x = cell % width, y = cell / width;
    int best = cell;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
          continue;
        const int neighbor = ny * width + nx;
        if (value(neighbor) > value(best))
          best = neighbor;
      }
    basin[cell] = best == cell ? cell : basin[best];
    if (best == cell) {
      const cv::Point2f center =
          origin + cv::Point2f{(x + 0.5f) * kCellM, (y + 0.5f) * kCellM};
      float closest = 4.0f;
      for (std::size_t id = 0; id < canonical_positions_.size(); ++id) {
        const cv::Vec2f delta{
            center.x - static_cast<float>(canonical_positions_[id].X().value()),
            center.y -
                static_cast<float>(canonical_positions_[id].Y().value())};
        const auto inverse =
            (canonical_covariances_[id] + cv::Matx22f::eye() * 0.16f).inv();
        const float distance = delta.dot(inverse * delta);
        if (distance < closest) {
          closest = distance;
          prior[cell] = id;
        }
      }
    }
  }
  struct Moments {
    float mass = 0;
    cv::Vec2f sum{};
    cv::Matx22f square{};
  };
  std::vector<Moments> moments(count);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (basin[cells[i]] < 0)
      continue;
    auto& m = moments[basin[cells[i]]];
    const float w = std::max(refined_weights[i], 0.1f * ground_weights[i]);
    const cv::Vec2f p{ground_points[i].x, ground_points[i].y};
    m.mass += w;
    m.sum += w * p;
    m.square += w * (p * p.t());
  }
  for (int cell : order) {
    if (basin[cell] < 0)
      continue;
    const int x = cell % width, y = cell / width;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
          continue;
        const int neighbor = ny * width + nx;
        if (basin[neighbor] < 0)
          continue;
        const auto mode_a = modes.Find(basin[cell]);
        const auto mode_b = modes.Find(basin[neighbor]);
        if (mode_a == mode_b)
          continue;
        const int peak_cell_a = peak_cell[mode_a];
        const int peak_cell_b = peak_cell[mode_b];
        const float separation =
            kCellM * std::hypot(peak_cell_a % width - peak_cell_b % width,
                                peak_cell_a / width - peak_cell_b / width);
        // A new density mode beyond an established pile's 2-sigma footprint
        // must not inflate that pile merely because its visible mass is small.
        const bool different_piles =
            prior[mode_a] != prior[mode_b] &&
            (prior[mode_a] >= 0 || prior[mode_b] >= 0) &&
            separation >= 2 * kBandwidthM;
        const auto moments_a = moments[mode_a];
        const auto moments_b = moments[mode_b];
        bool overlap = moments_a.mass == 0 || moments_b.mass == 0 ||
                       (prior[mode_a] >= 0 && prior[mode_a] == prior[mode_b]);
        if (!overlap) {
          const auto mean_a = moments_a.sum / moments_a.mass;
          const auto mean_b = moments_b.sum / moments_b.mass;
          const auto mean_delta = mean_a - mean_b;
          const auto covariance_a =
              moments_a.square / moments_a.mass - mean_a * mean_a.t() +
              cv::Matx22f::eye() * (kFuelRadiusM * kFuelRadiusM);
          const auto covariance_b =
              moments_b.square / moments_b.mass - mean_b * mean_b.t() +
              cv::Matx22f::eye() * (kFuelRadiusM * kFuelRadiusM);
          // Compare footprints at the pile resolution, rather than pixel noise.
          const auto covariance =
              covariance_a + covariance_b +
              cv::Matx22f::eye() * (2 * (kBandwidthM * kBandwidthM -
                                         kFuelRadiusM * kFuelRadiusM));
          overlap =
              mean_delta.dot(covariance.inv() * mean_delta) <= kModeOverlapD2;
          if (!overlap) {
            // Distant sampling lobes of one elongated Gaussian can still belong
            // together. Compare their weighted classification likelihoods.
            const float combined_mass = moments_a.mass + moments_b.mass;
            const auto combined_mean =
                (moments_a.sum + moments_b.sum) / combined_mass;
            const auto merged =
                (moments_a.square + moments_b.square) / combined_mass -
                combined_mean * combined_mean.t() +
                cv::Matx22f::eye() * (kFuelRadiusM * kFuelRadiusM);
            const double gain =
                0.5 *
                    (combined_mass * std::log(cv::determinant(merged)) -
                     moments_a.mass * std::log(cv::determinant(covariance_a)) -
                     moments_b.mass * std::log(cv::determinant(covariance_b))) +
                moments_a.mass * std::log(moments_a.mass / combined_mass) +
                moments_b.mass * std::log(moments_b.mass / combined_mass);
            overlap = gain <= 0;
          }
        }
        // Resolve weak entrants at 30 cm, but require a valley at 40 cm to split
        // untracked piles: sampling bumps along a broad pile should merge.
        if (!different_piles && overlap &&
            std::min(merge_value(cell), merge_value(neighbor)) >=
                0.75f * std::min(merge_value(peak_cell_a),
                                 merge_value(peak_cell_b))) {
          const bool keep_mode_a_peak = peak[mode_a] >= peak[mode_b];
          const float merged_peak = std::max(peak[mode_a], peak[mode_b]);
          const int merged_cell = keep_mode_a_peak ? peak_cell_a : peak_cell_b;
          const int merged_prior =
              prior[mode_a] >= 0 ? prior[mode_a] : prior[mode_b];
          modes.Union(mode_a, mode_b);
          const auto merged_root = modes.Find(mode_a);
          peak[merged_root] = merged_peak;
          peak_cell[merged_root] = merged_cell;
          prior[merged_root] = merged_prior;
          moments[merged_root] = {moments_a.mass + moments_b.mass,
                                  moments_a.sum + moments_b.sum,
                                  moments_a.square + moments_b.square};
        }
      }
  }
  std::vector<kmeans_cluster_t> result;
  std::vector<std::vector<float>> raw_weights;
  std::vector<int> indices(count, -1);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (basin[cells[i]] < 0)
      continue;
    const int root = modes.Find(basin[cells[i]]);
    if (indices[root] < 0) {
      indices[root] = result.size();
      result.emplace_back();
      raw_weights.emplace_back();
    }
    auto& cluster = result[indices[root]];
    cluster.img_points.push_back(image_points[i]);
    cluster.ground_points.push_back(ground_points[i]);
    cluster.ground_weights.push_back(refined_weights[i]);
    raw_weights[indices[root]].push_back(ground_weights[i]);
    cluster.fuel_count += ground_weights[i];
  }
  for (std::size_t index = 0; index < result.size(); ++index) {
    auto& c = result[index];
    const float refined_mass =
        std::accumulate(c.ground_weights.begin(), c.ground_weights.end(), 0.0f);
    // Occluded piles may have enough verified support but few exposed arcs.
    if (std::max(c.fuel_count, refined_mass) < 1.5f) {
      c.img_points.clear();
      continue;
    }
    if (refined_mass < 0.25f * c.fuel_count)
      c.ground_weights = std::move(raw_weights[index]);
    else
      c.fuel_count = refined_mass;
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
    auto position =
        frc::Translation2d{units::meter_t{center.x}, units::meter_t{center.y}};
    if (robot_pose)
      position = (position - robot_pose->Translation())
                     .RotateBy(-robot_pose->Rotation());
    c.robot_relative_translation = position;
    // Store public samples and covariance in robot coordinates.
    if (robot_pose) {
      const float a = -robot_pose->Rotation().Radians().value();
      const cv::Matx22f rotation{std::cos(a), -std::sin(a), std::sin(a),
                                 std::cos(a)};
      c.ground_covar = rotation * c.ground_covar * rotation.t();
      for (auto& p : c.ground_points) {
        const auto q =
            rotation *
            cv::Vec2f{p.x - static_cast<float>(robot_pose->X().value()),
                      p.y - static_cast<float>(robot_pose->Y().value())};
        p = {q[0], q[1]};
      }
    }
  }
  std::erase_if(result, [](const auto& c) { return c.img_points.empty(); });
  return result;
}

void HSVClusterTrackerNode::AssignPersistentIds(
    hsv_clusters_t& clusters, const std::optional<frc::Pose2d>& robot_pose) {
  std::vector<frc::Translation2d> positions;
  std::vector<std::tuple<double, std::size_t, std::size_t>> matches;
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    const auto offset = *clusters[i].robot_relative_translation;
    positions.push_back(robot_pose ? robot_pose->Translation() +
                                         offset.RotateBy(robot_pose->Rotation())
                                   : offset);
    for (std::size_t id = 0; id < canonical_positions_.size(); ++id) {
      const double distance =
          positions.back().Distance(canonical_positions_[id]).value();
      if (distance <= kIdMatchToleranceM)
        matches.emplace_back(distance, i, id);
    }
  }
  std::vector<cv::Matx22f> covariances;
  const float angle =
      robot_pose ? robot_pose->Rotation().Radians().value() : 0.0f;
  const cv::Matx22f rotation{std::cos(angle), -std::sin(angle), std::sin(angle),
                             std::cos(angle)};
  for (const auto& cluster : clusters)
    covariances.push_back(rotation * cluster.ground_covar * rotation.t());
  std::ranges::sort(matches);
  std::vector<bool> used(canonical_positions_.size(), false),
      assigned(clusters.size(), false);
  for (const auto& [distance, i, id] : matches) {
    if (used[id] || assigned[i])
      continue;
    clusters[i].id = id;
    used[id] = assigned[i] = true;
    canonical_positions_[id] = positions[i];
    canonical_covariances_[id] = covariances[i];
  }
  for (std::size_t i = 0; i < clusters.size(); ++i)
    if (!assigned[i]) {
      clusters[i].id = canonical_positions_.size();
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
