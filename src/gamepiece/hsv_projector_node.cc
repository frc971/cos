#include "gamepiece/hsv_projector_node.h"
#include <algorithm>
#include <cmath>
#include <numbers>

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Translation2d.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "gamepiece/image_utils.h"
#include "utils/cv_geometry.h"

namespace gamepiece {

HSVProjectorNode::HSVProjectorNode(std::string_view input_path,
                                   std::string_view output_path,
                                   cv::Matx33d intrinsics,
                                   cv::Matx44d extrinsics,
                                   const cv::Vec<double, 5>& distortion_coeffs,
                                   const HSVHeatmapConfig& heatmap_config,
                                   std::string_view pose_path)  // NOLINT
    : input_path_(input_path),
      output_path_(output_path),
      pose_path_(pose_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<HSVHeatmap>(
          output_path_)}),
      intrinsics_(intrinsics),
      extrinsics_cv_(extrinsics),
      distortion_coeffs_(distortion_coeffs),
      heatmap_config_(heatmap_config) {
  CHECK_GT(heatmap_config_.resolution_m, 0);
  const cv::Point2d span = heatmap_config_.max_m - heatmap_config_.min_m;
  CHECK_GT(span.x, 0);
  CHECK_GT(span.y, 0);
  heatmap_size_ = {
      static_cast<int>(std::ceil(span.x / heatmap_config_.resolution_m)),
      static_cast<int>(std::ceil(span.y / heatmap_config_.resolution_m))};
  CHECK_GT(extrinsics(2, 3), ball_height_m)
      << "Camera must be above the floor-supported balls";
  if (!pose_path_.empty()) {
    dependencies_.emplace_back(pose_path_,
                               typeid(control_loop::ValueMessage<frc::Pose2d>));
  }
  extrinsics_cv_ = extrinsics;
  utils::ChangeBasis(extrinsics_cv_, utils::Basis::kWpiToCv);
}

auto HSVProjectorNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    std::unique_lock processing_lock(processing_mutex_);
    // Context publications are write-once. Either dependency can arrive first;
    // wait for both publications, including explicit nulls, before producing
    // an output. Avoid duplicate callbacks and concurrent GPU scratch access.
    if (context->Exists(output_path_) ||
        (!pose_path_.empty() &&
         (!context->Exists(input_path_) || !context->Exists(pose_path_)))) {
      return;
    }
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    const auto* pose_message =
        pose_path_.empty()
            ? nullptr
            : context->GetMessage<control_loop::ValueMessage<frc::Pose2d>>(
                  pose_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0 ||
        (!pose_path_.empty() && pose_message == nullptr)) {
      context->SetMessage(output_path_, nullptr);
      processing_lock.unlock();
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }
    CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_BGRI);
    CHECK_GE(frame->stride, static_cast<size_t>(frame->width) * 3U);
    const frc::Pose2d pose =
        pose_message == nullptr ? frc::Pose2d{} : pose_message->value;
    CHECK(std::isfinite(pose.X().value()) && std::isfinite(pose.Y().value()) &&
          std::isfinite(pose.Rotation().Radians().value()));
    HSVHeatmap heatmap{.resolution_m = heatmap_config_.resolution_m,
                       .min_m = heatmap_config_.min_m,
                       .max_m = heatmap_config_.max_m,
                       .weights = cv::Mat1d::zeros(heatmap_size_),
                       .pixel_counts = cv::Mat1i::zeros(heatmap_size_)};
    const cv::cuda::GpuMat image(frame->height, frame->width, CV_8UC3,
                                 frame->destination.channel[0], frame->stride);
    HSVThreshold(image, min_hue, max_hue, min_saturation, hsv_image_,
                 threshold_mask_, stream_);
    cv::Mat1b mask;
    threshold_mask_.download(mask, stream_);
    stream_.waitForCompletion();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    const cv::Matx44d robot_to_camera = extrinsics_cv_.inv();
    const frc::Translation2d camera_position{
        units::meter_t{extrinsics_cv_(2, 3)},
        units::meter_t{-extrinsics_cv_(0, 3)}};
    for (const auto& contour : contours) {
      const cv::Rect bounds = cv::boundingRect(contour);
      cv::Mat1b blob = cv::Mat1b::zeros(bounds.size());
      cv::drawContours(blob, std::vector<std::vector<cv::Point>>{contour}, 0,
                       cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(),
                       0, -bounds.tl());
      cv::bitwise_and(blob, mask(bounds), blob);
      std::vector<cv::Point> pixels;
      cv::findNonZero(blob, pixels);
      thresholded_points_.clear();
      for (const auto& pixel : pixels) {
        thresholded_points_.emplace_back(pixel + bounds.tl());
      }
      const auto bottom_columns =
          LowestPointsByColumn(thresholded_points_, frame->width);
      std::vector<cv::Point2d> lowest_points;
      lowest_points.reserve(thresholded_points_.size());
      for (const auto& point : thresholded_points_) {
        lowest_points.emplace_back(
            point.x, InterpolatedLowestY(bottom_columns, point.x));
      }
      cv::undistortPoints(thresholded_points_, thresholded_points_, intrinsics_,
                          distortion_coeffs_);
      cv::undistortPoints(lowest_points, lowest_points, intrinsics_,
                          distortion_coeffs_);
      for (std::size_t index = 0; index < thresholded_points_.size(); ++index) {
        const auto& point = thresholded_points_[index];
        const cv::Point2d& bottom = lowest_points[index];
        const auto ground =
            UndistortedPinholePointOffset(bottom, 0, extrinsics_cv_);
        if (!ground.has_value()) continue;
        const cv::Vec4d robot_to_ball_top =
            robot_to_camera *
            cv::Vec4d{extrinsics_cv_(0, 3) - ground->Y().value(),
                      -ball_height_m,
                      extrinsics_cv_(2, 3) + ground->X().value(), 1};
        const double image_height =
            bottom.y - robot_to_ball_top[1] / robot_to_ball_top[2];
        if (robot_to_ball_top[2] <= 0 || !std::isfinite(image_height) ||
            image_height <= 0) continue;
        const double height_fraction = (bottom.y - point.y) / image_height;
        if (height_fraction < 0.5) {
          continue;  // Omit the frontmost ball's lower half.
        }
        const double height = ball_height_m * std::min(height_fraction, 1.0);
        const auto projected = UndistortedPinholePointOffset(
            point, static_cast<float>(-height), extrinsics_cv_);
        if (projected.has_value()) {
          const auto location =
              pose.Translation() +
              (*projected + camera_position).RotateBy(pose.Rotation());
          const double x = location.X().value();
          const double y = location.Y().value();
          const double weight = PixelWeight(point, height);
          CHECK(std::isfinite(x) && std::isfinite(y));
          CHECK(std::isfinite(weight) && weight > 0);
          ++heatmap.projected_pixels;
          heatmap.projected_weight += weight;
          if (x < heatmap.min_m.x || x >= heatmap.max_m.x ||
              y < heatmap.min_m.y || y >= heatmap.max_m.y) {
            ++heatmap.outside_pixels;
            continue;
          }
          const int col = std::min(
              static_cast<int>((x - heatmap.min_m.x) / heatmap.resolution_m),
              heatmap_size_.width - 1);
          const int row = std::min(
              static_cast<int>((y - heatmap.min_m.y) / heatmap.resolution_m),
              heatmap_size_.height - 1);
          ++heatmap.pixel_counts(row, col);
          heatmap.weights(row, col) += weight;
          heatmap.in_bounds_weight += weight;
        }
      }
    }

    context->SetMessage(
        output_path_, std::make_shared<control_loop::ValueMessage<HSVHeatmap>>(
                          std::move(heatmap)));
    processing_lock.unlock();
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto HSVProjectorNode::PixelWeight(const cv::Point2d& point,
                                   double height) const -> double {
  constexpr double radius = ball_height_m / 2;
  const cv::Vec4d ray = extrinsics_cv_ * cv::Vec4d{point.x, point.y, 1, 0};
  const double depth = (-height - extrinsics_cv_(1, 3)) / ray[1];
  // Estimate the ball center above the projected floor location, in camera axes.
  const cv::Vec3d center{
      depth * point.x + (height - radius) * extrinsics_cv_(1, 0),
      depth * point.y + (height - radius) * extrinsics_cv_(1, 1),
      depth + (height - radius) * extrinsics_cv_(1, 2)};
  // A sphere's pinhole area is pi*r^2*fx*fy*|center|/center.z^3.
  // Invert half that area because the lower half of the first ball is omitted.
  return 2 * center[2] * center[2] * center[2] /
         (std::numbers::pi * radius * radius * intrinsics_(0, 0) *
          intrinsics_(1, 1) * cv::norm(center));
}

void HSVProjectorNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto HSVProjectorNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto HSVProjectorNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace gamepiece
