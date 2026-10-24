#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <array>
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

constexpr std::pair<int, int> hsv_color_range{18, 30};
constexpr int minimum_saturation{100};
constexpr float kFuelRadiusM = 0.075f;
constexpr float kCellM = 0.1f;
constexpr float kBandwidthM = 0.3f;
constexpr float kMergeBandwidthM = 0.4f;
constexpr float kModeOverlapD2 = 4.6051702f;  // 90% contour for a 2D Gaussian.
// Zero-intercept visible-ball calibrations: validation/temporal-density/.
constexpr float kResidualAreaBalls = 4.305261f, kResidualArcBalls = 4.437539f;
constexpr float kCountAnchor = .432812f, kCountArea = 1.423993f,
                kCountArc = 0.0f;
constexpr int kOutline = 32;
const auto kCircle = [] {
  std::array<cv::Vec2f, kOutline> points;
  for (int i = 0; i < kOutline; ++i) {
    const float a = 2 * std::numbers::pi_v<float> * i / kOutline;
    points[i] = {std::cos(a), std::sin(a)};
  }
  return points;
}();

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
    cv::Matx33d intrinsics, cv::Matx44d extrinsics, std::size_t image_size,
    std::string_view pose_path, DensityHistory density_history,
    bool residual_density, std::string_view pose_topic,
    nt::NetworkTableInstance instance)
    : temporal_density_(density_history != DensityHistory::kNone),
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

void HSVClusterTrackerNode::EnablePoseOutput(std::string_view output_path) {
  if (output_path.empty() || output_path == input_path_ ||
      output_path == output_path_ || output_path == pose_path_ ||
      !pose_output_path_.empty()) {
    throw std::invalid_argument("Tracker pose output requires a distinct path");
  }
  pose_output_path_ = output_path;
  publications_.push_back(control_loop::MessageDescriptor::Publication<
      localization::PositionEstimateMessage>(pose_output_path_));
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
      image.download(frame_);
      ProcessFrame(frame_, pose);
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
    const cv::Mat& frame, std::optional<frc::Pose2d> robot_pose) {
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
    if (px >= -margin && px < frame.cols + margin && py >= -margin &&
        py < frame.rows + margin)
      ++track.missed;
  }
  std::erase_if(ball_tracks_, [](const auto& t) { return t.missed > 8; });
  for (auto& state : density_states_)
    ++state.missed;
  last_robot_pose_ = robot_pose;
  clusters_.clear();
  if (frame.empty())
    return;
  std::vector<cv::Point2f> image_points, ground_points;
  std::vector<float> ground_weights, refined_weights;
  cv::cvtColor(frame, hsv_image_, cv::COLOR_BGR2HSV);
  cv::inRange(hsv_image_,
              cv::Scalar{hsv_color_range.first, minimum_saturation, 0},
              cv::Scalar{hsv_color_range.second, 255, 255}, hsv_masked_);
  if (!cv::countNonZero(hsv_masked_))
    return;
  // Every silhouette tangent constrains a sphere center, without separating balls.
  // For unit ray u and inward perpendicular v, center = t*u + radius*v.
  // Its known height determines t. Occluded boundary arcs still provide evidence.
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(hsv_masked_, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);
  const cv::Matx33f rotation = camera_extrinsics_cv_.get_minor<3, 3>(0, 0);
  cv::Mat1b near_inside, near_outside;
  DilateTwoPixelL2(hsv_masked_, near_inside);
  DilateTwoPixelL2(255 - hsv_masked_, near_outside);
  // Verify tangent-derived sphere centers against the entire visible outline.
  // This also estimates the visible perimeter fraction for occlusion weighting.
  const auto visible_fraction = [&](const cv::Vec3f& center) {
    const float camera_to_center_distance = cv::norm(center);
    const auto camera_to_center = center / camera_to_center_distance;
    auto outline_basis_u = camera_to_center.cross(cv::Vec3f{0, 1, 0});
    outline_basis_u /= cv::norm(outline_basis_u);
    const auto outline_basis_v = camera_to_center.cross(outline_basis_u);
    const float sine_of_angular_radius =
        kFuelRadiusM / camera_to_center_distance;
    const float cosine_of_angular_radius =
        std::sqrt(1 - sine_of_angular_radius * sine_of_angular_radius);
    int supported = 0, accessible = 0, hidden = 0;
    for (int j = 0; j < kOutline; ++j) {
      const auto camera_center_to_sphere_cone_intersection =
          cosine_of_angular_radius * camera_to_center +
          sine_of_angular_radius * (kCircle[j][0] * outline_basis_u +
                                    kCircle[j][1] * outline_basis_v);
      const float x = camera_center_to_sphere_cone_intersection[0] /
                      camera_center_to_sphere_cone_intersection[2],
                  y = camera_center_to_sphere_cone_intersection[1] /
                      camera_center_to_sphere_cone_intersection[2];
      const int px =
          cvRound(x * camera_intrinsics_(0, 0) + camera_intrinsics_(0, 2));
      const int py =
          cvRound(y * camera_intrinsics_(1, 1) + camera_intrinsics_(1, 2));
      if (px < 0 || px >= frame.cols || py < 0 || py >= frame.rows)
        continue;
      ++accessible;
      if (near_inside(py, px) && near_outside(py, px))
        ++supported;
      if (!near_outside(py, px))
        ++hidden;
    }
    // Clipped arcs cannot contradict a candidate; verify the accessible outline.
    // Only arcs inside yellow or outside the image can be occluded. An arc
    // outside yellow is a fitting error and must not increase estimated mass.
    constexpr int minimum_supported_points = 4;
    constexpr float minimum_supported_to_accessible_ratio = 0.25;
    return supported >= minimum_supported_points &&
                   supported >=
                       minimum_supported_to_accessible_ratio * accessible
               ? static_cast<float>(accessible - hidden) / kOutline
               : 0.0f;
  };
  for (const auto& contour : contours) {
    constexpr size_t minimum_contour_track_length = 12;
    constexpr float minimum_contour_area = 8;
    if (contour.size() < minimum_contour_track_length ||
        cv::contourArea(contour) < minimum_contour_area)
      continue;
    std::vector<cv::Point2f> normalized;
    normalized.reserve(contour.size());
    for (const auto& p : contour)
      normalized.emplace_back(
          (p.x - camera_intrinsics_(0, 2)) / camera_intrinsics_(0, 0),
          (p.y - camera_intrinsics_(1, 2)) / camera_intrinsics_(1, 1));
    std::vector<cv::Vec3f> rays;
    for (const auto& p : normalized) {
      const cv::Vec3f ray{p.x, p.y, 1};
      rays.push_back(ray / cv::norm(ray));
    }
    const auto clipped = [&](std::size_t i) {
      const auto& p = contour[i];
      return p.x < 1 || p.x >= frame.cols - 1 || p.y < 1 ||
             p.y >= frame.rows - 1;
    };
    for (std::size_t i = 0; i < rays.size(); ++i) {
      // A contour along the sensor boundary is not a sphere tangent.
      const size_t before = (i + rays.size() - 4) % rays.size(),
                   after = (i + 4) % rays.size();
      if (clipped(i) || clipped(before) || clipped(after))
        continue;
      const cv::Vec3f tangent = rays[after] - rays[before];
      auto normal = tangent.cross(rays[i]);
      const float length = cv::norm(normal);
      if (length < 1e-6f)
        continue;
      const cv::Vec3f camera_normal = normal / length;
      normal = rotation * camera_normal;
      const cv::Vec3f ray = rotation * rays[i];
      if (ray[1] <= 1e-5f)
        continue;
      const float distance = (-kFuelRadiusM - camera_extrinsics_cv_(1, 3) -
                              kFuelRadiusM * normal[1]) /
                             ray[1];
      if (distance <= 0)
        continue;
      const float fraction =
          visible_fraction(distance * rays[i] + kFuelRadiusM * camera_normal);
      const cv::Vec3f camera_relative_center =
          distance * ray + kFuelRadiusM * normal;
      const frc::Translation2d offset =
          frc::Translation2d{units::meter_t{camera_relative_center[2]},
                             units::meter_t{-camera_relative_center[0]}} +
          extrinsics_offset_;
      const frc::Translation2d position =
          robot_pose ? robot_pose->Translation() +
                           offset.RotateBy(robot_pose->Rotation())
                     : offset;
      const cv::Point2f ground{static_cast<float>(position.X().value()),
                               static_cast<float>(position.Y().value())};
      if (robot_pose && (ground.x < 4.6f || ground.x > 11.94f ||
                         ground.y < 0.0f || ground.y > 8.07f))
        continue;
      // Inverse angular perimeter balances near/far silhouettes; hidden arcs
      // contribute no mass, so fuel_count estimates visible support, not inventory.
      const float weight =
          distance /
          (2 * std::numbers::pi_v<float> * kFuelRadiusM *
           std::sqrt(camera_intrinsics_(0, 0) * camera_intrinsics_(1, 1)));
      image_points.push_back(normalized[i]);
      ground_points.push_back(ground);
      ground_weights.push_back(weight);
      refined_weights.push_back(fraction > 0 ? weight / fraction : 0.0f);
    }
  }
  if (ground_points.empty())
    return;
  clusters_ = ClusterGroundPoints(image_points, ground_points, ground_weights,
                                  refined_weights, robot_pose);
  AssignPersistentIds(clusters_, robot_pose);
  EstimateDensities(frame, clusters_, robot_pose);
}

auto HSVClusterTrackerNode::SphereArcWeight(const cv::Matx33f& intrinsics,
                                            const cv::Vec3f& ray,
                                            const cv::Vec2f& normal,
                                            float depth) -> float {
  const float fx = intrinsics(0, 0), fy = intrinsics(1, 1);
  const float a = fx * normal[0], b = fy * normal[1],
              radial = a * ray[0] + b * ray[1];
  return depth * std::sqrt(a * a + b * b + radial * radial) /
         (2 * std::numbers::pi_v<float> * kFuelRadiusM * fx * fy *
          cv::norm(ray));
}

void HSVClusterTrackerNode::EstimateDensities(
    const cv::Mat& frame, hsv_clusters_t& clusters,
    const std::optional<frc::Pose2d>& robot_pose) {
  if (clusters.empty())
    return;
  auto intrinsics = camera_intrinsics_;
  float focal = std::sqrt(intrinsics(0, 0) * intrinsics(1, 1));
  const float height = -camera_extrinsics_cv_(1, 3) - kFuelRadiusM;
  const auto rotation = camera_extrinsics_cv_.get_minor<3, 3>(0, 0);
  const cv::Vec2f camera{static_cast<float>(extrinsics_offset_.X().value()),
                         static_cast<float>(extrinsics_offset_.Y().value())};
  cv::Mat image = frame, gray;
  cv::Mat1b mask = hsv_masked_, edges;
  auto crop = cv::boundingRect(mask);
  if (crop.empty())
    return;
  crop =
      cv::Rect{crop.x - 64, crop.y - 64, crop.width + 128, crop.height + 128} &
      cv::Rect{0, 0, frame.cols, frame.rows};
  image = image(crop);
  mask = mask(crop);
  intrinsics(0, 2) -= crop.x;
  intrinsics(1, 2) -= crop.y;
  float minimum_radius = std::numeric_limits<float>::max();
  for (const auto& c : clusters) {
    const auto& p = *c.robot_relative_translation;
    const auto position =
        rotation.t() *
        (cv::Vec3f{-static_cast<float>(p.Y().value()), -kFuelRadiusM,
                   static_cast<float>(p.X().value())} -
         camera_position_);
    if (position[2] > kFuelRadiusM)
      minimum_radius =
          std::min(minimum_radius, focal * kFuelRadiusM / position[2]);
  }
  // Normalize small apparent balls before edge voting; a fixed pixel stencil
  // otherwise discards their partially visible arcs as camera distance grows.
  const float scale = std::clamp(12.0f / minimum_radius, 1.0f, 3.0f);
  if (scale > 1) {
    cv::resize(image, image, {}, scale, scale, cv::INTER_LINEAR);
    cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);
    for (int row = 0; row < 2; ++row)
      for (int col = 0; col < 3; ++col)
        intrinsics(row, col) *= scale;
    focal *= scale;
  }
  cv::Mat1b explained(mask.size(), uchar{0}),
      explained_edges(mask.size(), uchar{0});
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  gray.setTo(0, mask == 0);
  cv::GaussianBlur(gray, gray, {3, 3}, 0.7);
  cv::Canny(gray, edges, 20, 40);
  cv::Mat1f gx, gy, edge_distance;
  cv::Mat1b near_mask;
  cv::Sobel(gray, gx, CV_32F, 1, 0);
  cv::Sobel(gray, gy, CV_32F, 0, 1);
  cv::Mat1i labels;
  cv::distanceTransform(255 - edges, edge_distance, labels, cv::DIST_L2, 5,
                        cv::DIST_LABEL_PIXEL);
  DilateTwoPixelL2(mask, near_mask);
  std::vector<cv::Vec2f> edge_normals(cv::countNonZero(edges) + 1);
  std::vector<cv::Point> edge_pixels(edge_normals.size());
  cv::Mat1f votes(mask.size(), 0.0f);
  const auto ray_at = [&](float x, float y) {
    return cv::Vec3f{(x - intrinsics(0, 2)) / intrinsics(0, 0),
                     (y - intrinsics(1, 2)) / intrinsics(1, 1), 1};
  };
  const auto arc_weight = [&](float depth, const cv::Vec3f& ray,
                              const cv::Vec2f& normal) {
    return SphereArcWeight(intrinsics, ray, normal, depth);
  };
  for (int y = 0; y < edges.rows; ++y) {
    const auto* edges_row = edges[y];
    const auto* gx_row = gx[y];
    const auto* gy_row = gy[y];
    const auto* labels_row = labels[y];
    for (int x = 0; x < edges.cols; ++x)
      if (edges_row[x]) {
        const float magnitude = std::hypot(gx_row[x], gy_row[x]);
        if (magnitude < 1e-5f)
          continue;
        edge_normals[labels_row[x]] = {gx_row[x] / magnitude,
                                       gy_row[x] / magnitude};
        edge_pixels[labels_row[x]] = {x, y};
      }
  }
  struct Sphere {
    cv::Point2f pixel;
    cv::Vec2f ground;
    float score = -1;
    std::array<cv::Point, kOutline> outline;
    std::array<int, kOutline> evidence_labels{};
    std::array<float, kOutline> evidence_support{};
    std::size_t owner{}, track = std::numeric_limits<std::size_t>::max();
    float depth = 0;
  };
  const auto score = [&](cv::Point2f pixel) {
    Sphere sphere;
    sphere.pixel = pixel;
    const auto ray = ray_at(pixel.x, pixel.y), ground_ray = rotation * ray;
    if (ground_ray[1] <= 1e-5f)
      return sphere;
    const auto center = height / ground_ray[1] * ray;
    if (center[2] <= kFuelRadiusM)
      return sphere;
    sphere.depth = center[2];
    sphere.ground = camera + cv::Vec2f{ground_ray[2], -ground_ray[0]} *
                                 (height / ground_ray[1]);
    const auto camera_to_center = center / cv::norm(center);
    auto outline_basis_u = camera_to_center.cross(cv::Vec3f{0, 1, 0});
    outline_basis_u /= cv::norm(outline_basis_u);
    const auto outline_basis_v = camera_to_center.cross(outline_basis_u);
    const float sine_of_angular_radius = kFuelRadiusM / cv::norm(center);
    const float cosine_of_angular_radius =
        std::sqrt(1 - sine_of_angular_radius * sine_of_angular_radius);
    std::array<cv::Point2f, kOutline> outline;
    for (int i = 0; i < kOutline; ++i) {
      const auto direction =
          cosine_of_angular_radius * camera_to_center +
          sine_of_angular_radius * (kCircle[i][0] * outline_basis_u +
                                    kCircle[i][1] * outline_basis_v);
      outline[i] = {
          direction[0] / direction[2] * intrinsics(0, 0) + intrinsics(0, 2),
          direction[1] / direction[2] * intrinsics(1, 1) + intrinsics(1, 2)};
      sphere.outline[i] = {cvRound(outline[i].x), cvRound(outline[i].y)};
    }
    float support = 0;
    int accessible = 0, bad = 0;
    for (int i = 0; i < kOutline; ++i) {
      const auto p = sphere.outline[i];
      if (p.x < 0 || p.x >= edges.cols || p.y < 0 || p.y >= edges.rows) {
        if (p.x / scale + crop.x >= 0 && p.x / scale + crop.x < frame.cols &&
            p.y / scale + crop.y >= 0 && p.y / scale + crop.y < frame.rows) {
          ++accessible;
          ++bad;
        }
        continue;
      }
      ++accessible;
      bad += !near_mask(p.y, p.x);
      const auto tangent =
          outline[(i + 1) % kOutline] - outline[(i + kOutline - 1) % kOutline];
      const auto normal = edge_normals[labels(p.y, p.x)];
      const float agreement = (normal[0] * -tangent.y + normal[1] * tangent.x) /
                              std::max(1e-5, cv::norm(tangent));
      const float distance = edge_distance(p.y, p.x) / 1.5f;
      sphere.evidence_labels[i] = labels(p.y, p.x);
      sphere.evidence_support[i] =
          std::exp(-0.5f * distance * distance) * agreement * agreement;
      support += sphere.evidence_support[i];
    }
    if (accessible >= 8)
      sphere.score = (support - 0.25f * bad) / accessible;
    return sphere;
  };
  std::vector<Sphere> candidates;
  const auto to_canonical = [&](cv::Point2f p) {
    const frc::Translation2d point{units::meter_t{p.x}, units::meter_t{p.y}};
    const auto q = robot_pose ? robot_pose->Translation() +
                                    point.RotateBy(robot_pose->Rotation())
                              : point;
    return cv::Point2f{static_cast<float>(q.X().value()),
                       static_cast<float>(q.Y().value())};
  };
  const auto to_robot = [&](cv::Point2f p) {
    const frc::Translation2d point{units::meter_t{p.x}, units::meter_t{p.y}};
    const auto q = robot_pose ? (point - robot_pose->Translation())
                                    .RotateBy(-robot_pose->Rotation())
                              : point;
    return cv::Vec2f{static_cast<float>(q.X().value()),
                     static_cast<float>(q.Y().value())};
  };
  cv::Mat1b predicted_edges(mask.size(), uchar{0});
  // Predict stationary field-space hypotheses before looking for new balls.
  if (predictive_density_)
    for (std::size_t i = 0; i < ball_tracks_.size(); ++i) {
      const auto ground = to_robot(ball_tracks_[i].position);
      const auto center =
          rotation.t() *
          (cv::Vec3f{-ground[1], -kFuelRadiusM, ground[0]} - camera_position_);
      if (center[2] <= kFuelRadiusM)
        continue;
      const cv::Point2f pixel{
          center[0] / center[2] * intrinsics(0, 0) + intrinsics(0, 2),
          center[1] / center[2] * intrinsics(1, 1) + intrinsics(1, 2)};
      const float margin = 2 * focal * kFuelRadiusM / center[2];
      if (pixel.x < -margin || pixel.x >= mask.cols + margin ||
          pixel.y < -margin || pixel.y >= mask.rows + margin)
        continue;
      Sphere best = score(pixel);
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0)
            continue;
          auto trial = score(pixel + cv::Point2f{static_cast<float>(dx),
                                                 static_cast<float>(dy)});
          if (trial.score > best.score)
            best = std::move(trial);
        }
      best.track = i;
      if (best.score < .20f)
        continue;  // History never substitutes for current evidence.
      const std::vector<cv::Point> outline(best.outline.begin(),
                                           best.outline.end());
      cv::polylines(predicted_edges, outline, true, 255, 3);
      candidates.push_back(std::move(best));
    }
  // Birth proposals use edges not explained by supported predictions.
  for (int y = 0; y < edges.rows; ++y) {
    const auto* edges_row = edges[y];
    const auto* predicted_edges_row = predicted_edges[y];
    const auto* gx_row = gx[y];
    const auto* gy_row = gy[y];
    const auto* labels_row = labels[y];
    for (int x = 0; x < edges.cols; ++x) {
      if (!edges_row[x] || predicted_edges_row[x] ||
          std::hypot(gx_row[x], gy_row[x]) < 1e-5f)
        continue;
      auto ray = ray_at(x, y);
      ray /= cv::norm(ray);
      const cv::Vec3f tangent{-gy_row[x] / intrinsics(0, 0),
                              gx_row[x] / intrinsics(1, 1), 0};
      auto normal = tangent.cross(ray);
      normal /= cv::norm(normal);
      const auto ground_ray = rotation * ray, ground_normal = rotation * normal;
      if (ground_ray[1] <= 1e-5f)
        continue;
      for (const int sign : {-1, 1}) {
        const float distance =
            (height - sign * kFuelRadiusM * ground_normal[1]) / ground_ray[1];
        if (distance <= 0)
          continue;
        const auto center = distance * ray + sign * kFuelRadiusM * normal;
        const int px = cvRound(center[0] / center[2] * intrinsics(0, 0) +
                               intrinsics(0, 2));
        const int py = cvRound(center[1] / center[2] * intrinsics(1, 1) +
                               intrinsics(1, 2));
        if (px >= 0 && px < votes.cols && py >= 0 && py < votes.rows)
          votes(py, px) +=
              arc_weight(center[2], ray_at(x, y), edge_normals[labels_row[x]]);
      }
    }
  }
  cv::Mat1f smooth;
  cv::GaussianBlur(votes, smooth, {}, 1.25);
  for (int y = 1; y + 1 < votes.rows; ++y) {
    const auto* smooth_row = smooth[y];
    for (int x = 1; x + 1 < votes.cols; ++x) {
      if (smooth_row[x] <
          0.05f / (2 * std::numbers::pi_v<float> * 1.25f * 1.25f))
        continue;
      bool peak = true;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          if (smooth(y + dy, x + dx) > smooth_row[x])
            peak = false;
      if (!peak)
        continue;
      Sphere best;
      for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
          auto candidate =
              score({static_cast<float>(x + dx), static_cast<float>(y + dy)});
          if (candidate.score > best.score)
            best = std::move(candidate);
        }
      if (best.score >= 0.20f)
        candidates.push_back(std::move(best));
    }
  }
  std::vector<cv::Vec2f> centers;
  std::vector<cv::Matx22f> inverse_footprints;
  std::vector<float> cluster_depths;
  for (const auto& c : clusters) {
    const auto& p = *c.robot_relative_translation;
    centers.emplace_back(p.X().value(), p.Y().value());
    inverse_footprints.push_back(
        (c.ground_covar + cv::Matx22f::eye() * 0.03f).inv());
    const auto& point = centers.back();
    cluster_depths.push_back(
        (rotation.t() * (cv::Vec3f{-point[1], -kFuelRadiusM, point[0]} -
                         camera_position_))[2]);
  }
  const auto owner = [&](const cv::Vec2f& point, float limit) {
    std::size_t cluster_index = clusters.size();
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      const auto d = point - centers[i];
      const float distance = d.dot(inverse_footprints[i] * d);
      if (distance < limit) {
        limit = distance;
        cluster_index = i;
      }
    }
    return cluster_index;
  };
  std::ranges::sort(candidates, [](const auto& a, const auto& b) {
    const bool candidate_a_is_tracked =
        a.track != std::numeric_limits<std::size_t>::max();
    const bool candidate_b_is_tracked =
        b.track != std::numeric_limits<std::size_t>::max();
    return candidate_a_is_tracked != candidate_b_is_tracked
               ? candidate_a_is_tracked
               : a.score > b.score;
  });
  std::vector<Sphere> verified;
  cv::Mat1f evidence_depth(mask.size(), 0.0f);
  cv::Mat1i evidence_owner(mask.size(), -1);
  for (const auto& sphere : candidates) {
    const auto i = owner(sphere.ground, 16);
    if (i == clusters.size())
      continue;
    std::vector<cv::Point> polygon(sphere.outline.begin(),
                                   sphere.outline.end());
    const auto box =
        cv::boundingRect(polygon) & cv::Rect{0, 0, mask.cols, mask.rows};
    if (box.empty())
      continue;
    for (auto& p : polygon)
      p -= box.tl();
    cv::Mat1b footprint(box.size(), uchar{0});
    cv::fillConvexPoly(footprint, polygon, 255);
    cv::Mat1b select = footprint & ((evidence_depth(box) == 0) |
                                    (evidence_depth(box) > sphere.depth));
    evidence_depth(box).setTo(sphere.depth, select);
    evidence_owner(box).setTo(static_cast<int>(i), select);
  }
  for (auto& sphere : candidates) {
    const bool predicted =
        sphere.track != std::numeric_limits<std::size_t>::max();
    sphere.owner = owner(sphere.ground, 9);
    if (sphere.owner == clusters.size() ||
        std::ranges::any_of(verified, [&](const auto& other) {
          return cv::norm(sphere.pixel - other.pixel) < 2.5;
        }))
      continue;
    if (predicted)
      ++clusters[sphere.owner].predicted_ball_count;
    if (!predicted && sphere.score < .30f)
      continue;
    float unique = 0;
    for (int j = 0; j < kOutline; ++j) {
      const auto pixel = edge_pixels[sphere.evidence_labels[j]];
      if (!explained_edges(pixel.y, pixel.x))
        unique += sphere.evidence_support[j];
    }
    if (unique < (predicted ? 3.0f : 4.0f))
      continue;  // Each added ball must explain new image edges.
    if (predictive_density_) {
      if (predicted) {
        auto& track = ball_tracks_[sphere.track];
        const auto measurement =
            to_canonical({sphere.ground[0], sphere.ground[1]});
        track.position += (measurement - track.position) /
                          static_cast<float>(
                              std::min<std::size_t>(8, track.observations + 1));
        ++track.observations;
        track.missed = 0;
        sphere.ground = to_robot(track.position);
        ++clusters[sphere.owner].reused_ball_count;
      } else {
        const auto position =
            to_canonical({sphere.ground[0], sphere.ground[1]});
        if (std::ranges::none_of(ball_tracks_, [&](const auto& t) {
              return cv::norm(t.position - position) < .10f;
            }))
          ball_tracks_.push_back({position});
      }
    }
    ++clusters[sphere.owner].verified_ball_count;
    clusters[sphere.owner].verified_ball_centers.emplace_back(sphere.ground[0],
                                                              sphere.ground[1]);
    const std::vector<cv::Point> polygon(sphere.outline.begin(),
                                         sphere.outline.end());
    cv::fillConvexPoly(explained, polygon, 255);
    const auto bounds = cv::boundingRect(polygon);
    cv::polylines(explained_edges, polygon, true, 255,
                  std::min(bounds.width, bounds.height) < 20 ? 1 : 3);
    verified.push_back(std::move(sphere));
  }
  // A zero-intercept predictor uses only image evidence not already explained
  // by verified spheres. Partial balls contribute even without a complete arc;
  // spheres with no image evidence cannot add a correction.
  for (int y = 0; y < mask.rows; ++y) {
    const auto* mask_row = mask[y];
    const auto* edges_row = edges[y];
    const auto* evidence_owner_row = evidence_owner[y];
    const auto* evidence_depth_row = evidence_depth[y];
    const auto* explained_row = explained[y];
    const auto* explained_edges_row = explained_edges[y];
    const auto* labels_row = labels[y];
    for (int x = 0; x < mask.cols; ++x) {
      const bool area = mask_row[x], arc = edges_row[x];
      if (!area && !arc)
        continue;
      const auto ray = ray_at(x, y), ground_ray = rotation * ray;
      if (ground_ray[1] <= 1e-5f)
        continue;
      const auto plane_point =
          camera +
          cv::Vec2f{ground_ray[2], -ground_ray[0]} * (height / ground_ray[1]);
      const auto i = evidence_owner_row[x] >= 0
                         ? static_cast<std::size_t>(evidence_owner_row[x])
                         : owner(plane_point, 16);
      if (i == clusters.size())
        continue;
      const float depth =
          evidence_depth_row[x] > 0 ? evidence_depth_row[x] : cluster_depths[i];
      if (depth <= kFuelRadiusM)
        continue;
      auto& evidence = clusters[i].residual_ball_evidence;
      auto& total = clusters[i].total_ball_evidence;
      const float fx = intrinsics(0, 0), fy = intrinsics(1, 1),
                  norm = cv::norm(ray);
      if (area) {
        const float weight = depth * depth /
                             (std::numbers::pi_v<float> * kFuelRadiusM *
                              kFuelRadiusM * fx * fy * norm);
        total[0] += weight;
        if (!explained_row[x])
          evidence[0] += weight;
      }
      if (arc) {
        const auto normal = edge_normals[labels_row[x]];
        const float weight = arc_weight(depth, ray, normal);
        total[1] += weight;
        if (!explained_edges_row[x])
          evidence[1] += weight;
      }
    }
  }
  for (auto& c : clusters) {
    c.instantaneous_ball_count =
        residual_density_
            ? c.verified_ball_count +
                  kResidualAreaBalls * c.residual_ball_evidence[0] +
                  kResidualArcBalls * c.residual_ball_evidence[1]
            : std::max(static_cast<float>(c.verified_ball_count),
                       kCountAnchor * c.verified_ball_count +
                           kCountArea * c.total_ball_evidence[0] +
                           kCountArc * c.total_ball_evidence[1]);
    auto covariance = c.ground_covar;
    auto footprint = c.verified_ball_centers;
    if (predictive_density_)
      for (const auto& track : ball_tracks_) {
        if (track.missed == 0 || track.missed > 3)
          continue;
        const auto point = to_robot(track.position);
        if (owner(point, 9) == static_cast<std::size_t>(&c - clusters.data()))
          footprint.emplace_back(point[0], point[1]);
      }
    if (footprint.size() >= 3)
      covariance =
          (Covariance(footprint) * static_cast<float>(footprint.size()) +
           3 * covariance) /
          static_cast<float>(footprint.size() + 3);
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
