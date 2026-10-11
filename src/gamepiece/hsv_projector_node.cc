#include "gamepiece/hsv_projector_node.h"
#include <algorithm>
#include <cmath>

#include <frc/geometry/Translation2d.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "gamepiece/image_utils.h"
#include "utils/cv_geometry.h"

namespace gamepiece {

HSVProjectorNode::HSVProjectorNode(
    std::string_view input_path, std::string_view output_path,
    cv::Matx33d intrinsics, cv::Matx44d extrinsics,
    cv::Vec<double, 5> distortion_coeffs)  // NOLINT
    : input_path_(input_path),
      output_path_(output_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<
          std::vector<frc::Translation2d>>(output_path_)}),
      intrinsics_(intrinsics),
      extrinsics_cv_(extrinsics),
      distortion_coeffs_(distortion_coeffs) {
  cv::Mat camera_pose(extrinsics_cv_);
  utils::ChangeBasis(camera_pose, utils::Basis::kWpiToCv);
  extrinsics_cv_ = cv::Matx44d(camera_pose);
}

auto HSVProjectorNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      context->SetMessage(output_path_, nullptr);
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }
    CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_BGRI);
    CHECK_GE(frame->stride, static_cast<size_t>(frame->width) * 3U);
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
    std::vector<frc::Translation2d> projected_points;
    projected_points.reserve(static_cast<size_t>(cv::countNonZero(mask)));
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
      cv::undistortPoints(thresholded_points_, thresholded_points_, intrinsics_,
                          distortion_coeffs_);
      const cv::Point2d bottom =
          *std::ranges::max_element(thresholded_points_, {}, &cv::Point2d::y);
      const auto ground =
          UndistortedPinholePointOffset(bottom, 0, extrinsics_cv_);
      if (!ground.has_value()) {
        continue;
      }
      const cv::Vec4d robot_to_ball_top =
          robot_to_camera *
          cv::Vec4d{extrinsics_cv_(0, 3) - ground->Y().value(), -ball_height_m,
                    extrinsics_cv_(2, 3) + ground->X().value(), 1};
      const double image_height =
          bottom.y - robot_to_ball_top[1] / robot_to_ball_top[2];
      if (robot_to_ball_top[2] <= 0 || !std::isfinite(image_height) ||
          image_height <= 0) {
        continue;
      }
      for (const auto& point : thresholded_points_) {
        const double height =
            ball_height_m *
            std::clamp((bottom.y - point.y) / image_height, 0.0, 1.0);
        const auto projected = UndistortedPinholePointOffset(
            point, static_cast<float>(-height), extrinsics_cv_);
        if (projected.has_value()) {
          projected_points.push_back(*projected + camera_position);
        }
      }
    }

    context->SetMessage(
        output_path_,
        std::make_shared<
            control_loop::ValueMessage<std::vector<frc::Translation2d>>>(
            std::move(projected_points)));
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
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
